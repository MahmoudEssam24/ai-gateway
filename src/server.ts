import express from "express";
import cors from "cors";
import Groq from "groq-sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const PORT = Number(process.env.PORT ?? "3002");
const MCP_SERVER_URL = process.env.MCP_SERVER_URL ?? "https://mcp-server-production-b54a.up.railway.app/mcp"; 
const GROQ_API_KEY = process.env.GROQ_API_KEY;

// ✅ User Requirement: Keep the specific model
const GROQ_MODEL = process.env.GROQ_MODEL ?? "openai/gpt-oss-120b";

if (!GROQ_API_KEY) {
  console.error("❌ Error: GROQ_API_KEY is missing.");
  process.exit(1);
}

const app = express();
app.use(cors());
app.use(express.json());

const groq = new Groq({ apiKey: GROQ_API_KEY });

// --- State ---
let mcpClient: Client | null = null;
const conversationStore = new Map<string, Groq.Chat.ChatCompletionMessageParam[]>();
let cachedGroqTools: Groq.Chat.ChatCompletionTool[] | null = null;

// --- Helper Functions ---

async function getMcpClient(): Promise<Client> {
  if (mcpClient) return mcpClient;
  console.log(`🔌 Connecting to MCP Server at ${MCP_SERVER_URL}...`);
  const client = new Client({ name: "ai-gateway", version: "1.0.0" });
  const transport = new StreamableHTTPClientTransport(new URL(MCP_SERVER_URL));
  await client.connect(transport);
  mcpClient = client;
  console.log("✅ MCP Client Connected.");
  return client;
}

async function getGroqTools(): Promise<Groq.Chat.ChatCompletionTool[]> {
  if (cachedGroqTools) return cachedGroqTools;
  try {
    const client = await getMcpClient();
    const result = await client.listTools();
    cachedGroqTools = result.tools.map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema as Record<string, any>, 
      },
    }));
    return cachedGroqTools;
  } catch (error) {
    console.error("❌ Failed to fetch tools from MCP:", error);
    return []; 
  }
}

function extractTextFromMcpResult(result: any): string {
  if (result?.structuredContent) return JSON.stringify(result.structuredContent, null, 2);
  const firstText = result?.content?.find((c: any) => c.type === "text")?.text;
  if (typeof firstText === "string") return firstText;
  return JSON.stringify(result, null, 2);
}

// --- Main Route ---

app.post("/chat/openai", async (req, res) => {
  try {
    const { message, systemPrompt, conversationId } = req.body;

    if (!conversationId) return res.status(400).json({ error: "conversationId is required" });

    // 1. History & System Prompt
    let messages = conversationStore.get(conversationId);
    if (!messages) {
      messages = [];
      
      // ✅ FIX 1: Strict System Instruction
      // We explicitly tell the model NOT to write text-based tool calls.
      const forcedSystemPrompt = `You are a helpful assistant.
IMPORTANT: You have access to tools.
- If you need to use a tool, you must generate a formal Tool Call event.
- DO NOT write the tool call as text in your response (e.g. "[get_user_info(...)]").
- DO NOT describe what you are doing. Just trigger the tool.
${systemPrompt || ""}`;

      messages.push({ role: "system", content: forcedSystemPrompt });
    }

    messages.push({ role: "user", content: message });
    const availableTools = await getGroqTools();

    // 2. Call Groq
    const runner = await groq.chat.completions.create({
      model: GROQ_MODEL,
      messages: messages,
      tools: availableTools.length > 0 ? availableTools : undefined,
      tool_choice: availableTools.length > 0 ? "auto" : "none",
      
      // ✅ FIX 2: Lower Temperature (Crucial for Llama 4 Scout)
      // This stops it from being "creative" with the tool format.
      temperature: 0.1, 
      
      // ✅ FIX 3: Disable Parallel Calls (Improves stability)
      parallel_tool_calls: false,
      
      max_completion_tokens: 1024,
    });

    const responseMessage = runner.choices[0].message;

    // 3. Handle Tool Execution
    if (responseMessage.tool_calls && responseMessage.tool_calls.length > 0) {
      console.log(`🤖 Executing ${responseMessage.tool_calls.length} tool(s)...`);
      messages.push(responseMessage);
      
      const client = await getMcpClient();

      for (const toolCall of responseMessage.tool_calls) {
        const toolName = toolCall.function.name;
        const toolArgsString = toolCall.function.arguments;

        try {
          const args = JSON.parse(toolArgsString);
          const mcpResult = await client.callTool({
            name: toolName,
            arguments: args,
          });

          messages.push({
            tool_call_id: toolCall.id,
            role: "tool",
            content: extractTextFromMcpResult(mcpResult),
          });
        } catch (error: any) {
          console.error(`❌ Tool Error (${toolName}):`, error.message);
          messages.push({
            tool_call_id: toolCall.id,
            role: "tool",
            content: JSON.stringify({ error: error.message }),
          });
        }
      }

      // 4. Final Answer
      const finalResponse = await groq.chat.completions.create({
        model: GROQ_MODEL,
        messages: messages,
        // Keep low temp for the final answer too
        temperature: 0.3, 
      });

      const finalContent = finalResponse.choices[0].message;
      messages.push(finalContent);
      conversationStore.set(conversationId, messages);

      return res.json({
        conversationId,
        text: finalContent.content,
      });
    }

    // No tools used
    messages.push(responseMessage);
    conversationStore.set(conversationId, messages);

    res.json({
      conversationId,
      text: responseMessage.content,
    });

  } catch (err: any) {
    console.error("Gateway Error:", err);
    // Return detailed error to help debugging
    res.status(500).json({ error: err?.message || "Internal Server Error", details: err });
  }
});

app.get("/health", (_, res) => res.json({ ok: true, mcp: !!mcpClient }));

app.listen(PORT, async () => {
  console.log(`🚀 AI Gateway running on port ${PORT}`);
  console.log(`🧠 Model: ${GROQ_MODEL}`);
  await getGroqTools(); // Warm up tools
});