// ai-gateway/src/server.ts
import express from "express";
import cors from "cors";
import OpenAI from "openai";

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v || v.trim() === "") throw new Error(`Missing env var: ${name}`);
  return v;
}

const PORT = Number(process.env.PORT ?? "3002");

// MCP server URLs
const MCP_SERVER_URL =
  process.env.MCP_SERVER_URL ??
  "https://mcp-server-production-b54a.up.railway.app/mcp";

// =======================
// Cohere config (ACTIVE)
// =======================
const COHERE_API_KEY = requireEnv("COHERE_API_KEY");
const COHERE_MODEL = process.env.COHERE_MODEL ?? "command-a-03-2025";

// =======================
// Groq config (COMMENTED)
// =======================
// const GROQ_API_KEY = requireEnv("GROQ_API_KEY");
// const GROQ_MODEL =
//   process.env.GROQ_MODEL ?? "meta-llama/llama-4-scout-17b-16e-instruct";

// Optional default system prompt (you still override it per persona)
const DEFAULT_SYSTEM_PROMPT = `You are an assistant for a POC that automates backend actions via MCP tools.
Use the tools to help the user. Ask for missing data one-by-one and always ask before executing any action.`;

// Allowed tools in your MCP server (must match MCP server tools)
export const allowedTools = [
  "get_user_info",
  "create_parking_card",
  "create_vacation_request",
  "request_house_maid",
  "request_home_checkup",
  "list_assistive_devices",
  "submit_medical_device_aid_request",
  "get_medical_device_aid_request",
  "list_medical_device_aid_requests",
] as const;

export type AllowedToolName = (typeof allowedTools)[number];

const app = express();
app.use(cors());
app.use(express.json());

// =====================================================
// Cohere via OpenAI SDK (Compatibility API) (ACTIVE)
// Base URL per Cohere docs
// =====================================================
const cohere = new OpenAI({
  apiKey: COHERE_API_KEY,
  baseURL: "https://api.cohere.ai/compatibility/v1",
});

// =======================
// Groq client (COMMENTED)
// =======================
// const groq = new OpenAI({
//   apiKey: GROQ_API_KEY,
//   baseURL: "https://api.groq.com/openai/v1",
// });

// Simple chat history per conversationId
type Role = "system" | "user" | "assistant";
type ChatMessage = { role: Role; content: string };

const conversations = new Map<string, ChatMessage[]>();

// -----------------------------
// MCP JSON-RPC helpers (HTTP)
// -----------------------------
type JsonRpcReq = { jsonrpc: "2.0"; id: string | number; method: string; params?: any };
type JsonRpcRes = { jsonrpc: "2.0"; id: string | number; result?: any; error?: { code: number; message: string; data?: any } };

async function mcpRpc<T>(method: string, params?: any): Promise<T> {
  const req: JsonRpcReq = { jsonrpc: "2.0", id: Date.now(), method, params };

  const res = await fetch(MCP_SERVER_URL, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      // your MCP server patches Accept, but sending both helps
      accept: "application/json, text/event-stream",
    },
    body: JSON.stringify(req),
  });

  const text = await res.text().catch(() => "");
  if (!res.ok) throw new Error(`MCP HTTP ${res.status}: ${text || res.statusText}`);

  const data = (text ? JSON.parse(text) : {}) as JsonRpcRes;
  if (data.error) throw new Error(`MCP RPC error ${data.error.code}: ${data.error.message}`);
  return data.result as T;
}

type McpTool = {
  name: string;
  description?: string;
  inputSchema?: any; // JSON schema
};

async function listMcpTools(): Promise<McpTool[]> {
  const out = await mcpRpc<{ tools: McpTool[] }>("tools/list", {});
  return out?.tools ?? [];
}

async function callMcpTool(name: string, args: any): Promise<any> {
  return mcpRpc<any>("tools/call", { name, arguments: args ?? {} });
}

// -----------------------------
// Convert MCP tools → OpenAI tools
// (Cohere compatibility supports `tools` on chat.completions)
// -----------------------------
function toOpenAiTools(mcpTools: McpTool[]) {
  const allowed = new Set<string>(allowedTools);

  return mcpTools
    .filter((t) => allowed.has(t.name))
    .map((t) => ({
      type: "function" as const,
      function: {
        name: t.name,
        description: t.description ?? "",
        // MCP usually provides JSON schema in inputSchema; fallback to empty object
        parameters: t.inputSchema ?? { type: "object", properties: {} },
      },
    }));
}

// -----------------------------
// Extract tool result text from MCP response
// -----------------------------
function extractMcpText(result: any): string {
  // MCP tool result commonly: { content: [{type:"text", text:"..."}], structuredContent: ... }
  const content = result?.content;
  if (Array.isArray(content)) {
    const texts = content
      .map((c: any) => (c?.type === "text" ? String(c.text ?? "") : ""))
      .filter(Boolean);
    if (texts.length) return texts.join("\n");
  }
  // fallback
  if (typeof result?.structuredContent !== "undefined") {
    return JSON.stringify(result.structuredContent, null, 2);
  }
  return JSON.stringify(result ?? {}, null, 2);
}

/**
 * POST /chat/openai
 * Body: { conversationId?: string; systemPrompt?: string; message: string }
 *
 * Route stays `/chat/openai` so the frontend doesn’t change.
 *
 * Now:
 * - We call Cohere (command-a-03-2025 default) via Compatibility API
 * - We fetch MCP tools (tools/list), expose them as function tools
 * - If Cohere requests tool calls, we execute them by calling MCP (tools/call)
 * - Then we send tool results back to Cohere and continue until final answer
 */
app.post("/chat/openai", async (req, res) => {
  try {
    const { conversationId, systemPrompt, message } = req.body as {
      conversationId?: string;
      systemPrompt?: string;
      message: string;
    };

    const cid = conversationId ?? "default";

    // Get or initialize history
    let history = conversations.get(cid) ?? [];

    // First turn: add system prompt (persona prompt) or default
    if (history.length === 0) {
      const sys = (systemPrompt ?? DEFAULT_SYSTEM_PROMPT).trim();
      if (sys) history.push({ role: "system", content: sys });
    }

    // Add the new user message
    history.push({ role: "user", content: message });

    // Pull MCP tools and expose only allowed ones
    const mcpTools = await listMcpTools();
    const tools = toOpenAiTools(mcpTools);

    // Convert your history to OpenAI messages
    // (Compatibility API supports system/user/assistant; keep system as "system")
    const messages: any[] = history.map((m) => ({ role: m.role, content: m.content }));
    

    // Run a small tool loop (multi-step tool use)
    let finalText = "";
    const MAX_STEPS = 8;

    for (let step = 0; step < MAX_STEPS; step++) {
      const completion = await cohere.chat.completions.create({
        model: COHERE_MODEL,
        messages,
        tools,
        tool_choice: "auto",
      });

      const choice = completion.choices?.[0];
      const assistantMsg = choice?.message;
      console.log("Messages are: ", assistantMsg);
      const assistantText = (assistantMsg?.content as string) ?? "";
      const toolCalls = (assistantMsg as any)?.tool_calls ?? [];

      // If assistant returned normal text AND no tool calls => final
      if ((!toolCalls || toolCalls.length === 0) && assistantText) {
        finalText = assistantText;
        messages.push({ role: "assistant", content: assistantText });
        break;
      }

      // If tool calls exist, execute them via MCP server
      if (toolCalls && toolCalls.length > 0) {
        // Keep any assistant text (optional)
        messages.push({
          role: "assistant",
          content: assistantText || "",
          tool_calls: toolCalls,
        });

        for (const tc of toolCalls) {
          const fn = tc?.function;
          const toolName = fn?.name as AllowedToolName;
          const rawArgs = fn?.arguments ?? "{}";

          let argsObj: any = {};
          try {
            argsObj = typeof rawArgs === "string" ? JSON.parse(rawArgs) : rawArgs;
          } catch {
            argsObj = {};
          }

          // Execute MCP tool
          const toolResult = await callMcpTool(toolName, argsObj);
          const toolText = extractMcpText(toolResult);

          // Feed tool result back using OpenAI tool message format
          messages.push({
            role: "tool",
            tool_call_id: tc.id,
            content: toolText,
          });
        }

        // continue loop
        continue;
      }

      // If neither text nor tool calls, stop defensively
      finalText = assistantText || "";
      break;
    }

    // Save assistant message into history for next turn
    if (finalText) history.push({ role: "assistant", content: finalText });
    conversations.set(cid, history);

    res.json({ conversationId: cid, text: finalText });
  } catch (err: any) {
    console.error("Cohere /chat/openai error:", err);
    res.status(500).json({ error: err?.message || "Cohere Chat error" });
  }
});

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

app.listen(PORT, () => {
  console.log(`AI Gateway (Cohere) listening on http://localhost:${PORT}`);
  console.log(`COHERE_MODEL: ${COHERE_MODEL}`);
  console.log(`MCP_SERVER_URL: ${MCP_SERVER_URL}`);
});
