import express from "express";
import cors from "cors";
import Groq from "groq-sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";


// --- Configuration ---
const PORT = Number(process.env.PORT ?? "3002");
const MCP_SERVER_URL = process.env.MCP_SERVER_URL ?? "https://mcp-server-production-b54a.up.railway.app/mcp";
const GROQ_API_KEY = process.env.GROQ_API_KEY;
// The specific model you requested
const GROQ_MODEL = process.env.GROQ_MODEL ?? "meta-llama/llama-4-scout-17b-16e-instruct";

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
// Key: conversationId, Value: Array of messages
const conversationStore = new Map<string, Groq.Chat.ChatCompletionMessageParam[]>();

// 3. Tool Cache (Optional: to prevent fetching tools on every request)
let cachedGroqTools: Groq.Chat.ChatCompletionTool[] | null = null;


// --- Helper Functions ---

/**
 * Connects to the MCP Server (Streamable HTTP)
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
 * Fetches tools from MCP and converts them to Groq/OpenAI JSON Schema format
 */
async function getGroqTools(): Promise<Groq.Chat.ChatCompletionTool[]> {
  // If you want to refresh tools on every request (for dev), comment out the next line
  if (cachedGroqTools) return cachedGroqTools;

  try {
    const client = await getMcpClient();
    const result = await client.listTools();

    // Transform MCP tools to Groq tools
    cachedGroqTools = result.tools.map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        // MCP inputSchema is compatible with Groq parameters
        parameters: tool.inputSchema as Record<string, any>, 
      },
    }));

    console.log(`🛠️ Discovered ${cachedGroqTools.length} tools from MCP.`);
    return cachedGroqTools;
  } catch (error) {
    console.error("❌ Failed to fetch tools from MCP:", error);
    // Return empty array so the chat can still function (just without tools)
    return []; 
  }
}

/**
 * Helper to extract clean text from MCP tool results
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

    // 1. Load History or Initialize New
    let messages = conversationStore.get(conversationId);
    if (!messages) {
      messages = [];
      if (systemPrompt) {
        messages.push({ role: "system", content: systemPrompt });
      }
    }

    // 2. Add User Message
    messages.push({ role: "user", content: message });

    // 3. Get Tools (Dynamic)
    const availableTools = await getGroqTools();

    // 4. Initial Call to Groq
    const runner = await groq.chat.completions.create({
      model: GROQ_MODEL,
      messages: messages,
      tools: availableTools.length > 0 ? availableTools : undefined,
      tool_choice: availableTools.length > 0 ? "auto" : "none",
      temperature: 0.6, 
      max_completion_tokens: 1024,
    });

    const responseMessage = runner.choices[0].message;

    // 5. Handle Tool Execution Loop
    if (responseMessage.tool_calls && responseMessage.tool_calls.length > 0) {
      console.log(`🤖 Model requesting ${responseMessage.tool_calls.length} tool(s)...`);
      
      // A. Add Assistant's "Request" to history
      messages.push(responseMessage);

      const client = await getMcpClient();

      // B. Execute all requested tools
      for (const toolCall of responseMessage.tool_calls) {
        const toolName = toolCall.function.name;
        const toolArgsString = toolCall.function.arguments;

        console.log(`   > Executing: ${toolName} with args: ${toolArgsString}`);

        try {
          const args = JSON.parse(toolArgsString);

          // C. Call MCP
          const mcpResult = await client.callTool({
            name: toolName,
            arguments: args,
          });

          // D. Add "Success" result to history
          messages.push({
            tool_call_id: toolCall.id,
            role: "tool",
            content: extractTextFromMcpResult(mcpResult),
          });

        } catch (error: any) {
          console.error(`   ❌ Tool Error (${toolName}):`, error.message);
          
          // E. Add "Error" result to history (so Groq knows to retry or ask user)
          messages.push({
            tool_call_id: toolCall.id,
            role: "tool",
            content: JSON.stringify({ 
              error: `Execution failed: ${error.message}. Please check arguments or ask user for missing info.` 
            }),
          });
        }
      }

      // 6. Final Call to Groq (Synthesize answer from tool results)
      const finalResponse = await groq.chat.completions.create({
        model: GROQ_MODEL,
        messages: messages,
        // Usually, we don't pass tools in the final step to prevent infinite loops,
        // unless you want a multi-step agent.
      });

      const finalContent = finalResponse.choices[0].message;
      
      // Save final answer
      messages.push(finalContent);
      conversationStore.set(conversationId, messages);

      return res.json({
        conversationId,
        text: finalContent.content,
      });
    }

    // 7. No Tools Used (Just a conversational reply)
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
  console.log(`🔗 Linking to MCP Server: ${MCP_SERVER_URL}`);
  
  // Pre-load tools on startup to ensure connection works
  await getGroqTools();
});