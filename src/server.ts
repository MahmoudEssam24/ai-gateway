import express from "express";
import cors from "cors";
import Groq from "groq-sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

// --- Configuration ---
const PORT = Number(process.env.PORT ?? "3002");
const MCP_SERVER_URL = process.env.MCP_SERVER_URL ?? "https://mcp-server-production-b54a.up.railway.app/mcp"; 
const GROQ_API_KEY = process.env.GROQ_API_KEY;

// Use the specific model you mentioned
const GROQ_MODEL = process.env.GROQ_MODEL ?? "openai/gpt-oss-120b";

if (!GROQ_API_KEY) {
  console.error("❌ Error: GROQ_API_KEY is missing in environment variables.");
  process.exit(1);
}

const app = express();
app.use(cors());
app.use(express.json());

const groq = new Groq({ apiKey: GROQ_API_KEY });

// --- State Management ---
// 1. MCP Client Singleton
let mcpClient: Client | null = null;

// 2. In-Memory Conversation History
const conversationStore = new Map<string, Groq.Chat.ChatCompletionMessageParam[]>();

// 3. Tool Cache
let cachedGroqTools: Groq.Chat.ChatCompletionTool[] | null = null;

// --- Helper Functions ---

/**
 * Connects to the MCP Server
 */
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

/**
 * Fetches tools from MCP and converts them to Groq format
 */
async function getGroqTools(): Promise<Groq.Chat.ChatCompletionTool[]> {
  // Simple caching to avoid fetching on every request. 
  // Restart server to refresh tools.
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

    console.log(`🛠️ Discovered ${cachedGroqTools.length} tools from MCP.`);
    return cachedGroqTools;
  } catch (error) {
    console.error("❌ Failed to fetch tools from MCP:", error);
    return []; 
  }
}

/**
 * Clean helper to extract text from MCP results
 */
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

    if (!conversationId) {
      return res.status(400).json({ error: "conversationId is required" });
    }

    // 1. Load History
    let messages = conversationStore.get(conversationId);
    if (!messages) {
      messages = [];
      
      // Strict System Prompt to prevent "chatty" tool calls
      const defaultSystem = `You are a helpful assistant.
      IMPORTANT: If you need to use a tool, generate the tool call JSON silently. 
      Do NOT describe what you are doing (e.g. do not write "I will check...").
      Just execute the function.`;

      messages.push({ 
        role: "system", 
        content: systemPrompt ? `${defaultSystem}\n${systemPrompt}` : defaultSystem 
      });
    }

    // 2. Add User Message
    messages.push({ role: "user", content: message });

    // 3. Get Tools
    const availableTools = await getGroqTools();
    const hasTools = availableTools.length > 0;

    // 4. Initial Call to Groq
    const runner = await groq.chat.completions.create({
      model: GROQ_MODEL,
      messages: messages,
      // CRITICAL: Only pass tools/tool_choice if tools actually exist
      tools: hasTools ? availableTools : undefined,
      tool_choice: hasTools ? "auto" : "none",
      temperature: 0.1, // Keep low for reliable tool usage
      max_completion_tokens: 1024,
    });

    const responseMessage = runner.choices[0].message;

    // 5. Handle Tool Execution
    if (responseMessage.tool_calls && responseMessage.tool_calls.length > 0) {
      console.log(`🤖 Model requesting ${responseMessage.tool_calls.length} tool(s)...`);
      
      // Append assistant request to history
      messages.push(responseMessage);

      const client = await getMcpClient();

      for (const toolCall of responseMessage.tool_calls) {
        const toolName = toolCall.function.name;
        const toolArgsString = toolCall.function.arguments;

        console.log(`   > Executing: ${toolName}`);

        try {
          const args = JSON.parse(toolArgsString);

          const mcpResult = await client.callTool({
            name: toolName,
            arguments: args,
          });

          // Append success result
          messages.push({
            tool_call_id: toolCall.id,
            role: "tool",
            content: extractTextFromMcpResult(mcpResult),
          });

        } catch (error: any) {
          console.error(`   ❌ Tool Error (${toolName}):`, error.message);
          
          // Append error result so model can self-correct
          messages.push({
            tool_call_id: toolCall.id,
            role: "tool",
            content: JSON.stringify({ 
              error: `Tool execution failed: ${error.message}. Please check arguments.` 
            }),
          });
        }
      }

      // 6. Final Call to Groq (Synthesize answer)
      const finalResponse = await groq.chat.completions.create({
        model: GROQ_MODEL,
        messages: messages,
        // We usually do NOT pass tools in the final step to prevent infinite loops
        // unless you want multi-step agents.
        temperature: 0.5, 
      });

      const finalContent = finalResponse.choices[0].message;
      messages.push(finalContent);
      conversationStore.set(conversationId, messages);

      return res.json({
        conversationId,
        text: finalContent.content,
      });
    }

    // 7. No Tools Used (Standard Reply)
    messages.push(responseMessage);
    conversationStore.set(conversationId, messages);

    res.json({
      conversationId,
      text: responseMessage.content,
    });

  } catch (err: any) {
    console.error("Gateway Error:", err);
    res.status(500).json({ error: err?.message || "Internal Server Error" });
  }
});

// --- Health Check ---
app.get("/health", (_, res) => res.json({ 
  status: "ok", 
  mcpConnected: !!mcpClient,
  toolsLoaded: cachedGroqTools?.length ?? 0 
}));

app.listen(PORT, async () => {
  console.log(`🚀 AI Gateway listening on http://localhost:${PORT}`);
  console.log(`🔗 Target MCP URL: ${MCP_SERVER_URL}`);
  console.log(`🧠 Model: ${GROQ_MODEL}`);
  
  // Pre-load tools on startup
  await getGroqTools();
});