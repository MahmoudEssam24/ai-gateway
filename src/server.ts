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
// - MCP_SERVER_URL: default/internal URL (for your own reference)
// - GROQ_MCP_SERVER_URL: public URL that Groq can reach
const MCP_SERVER_URL =
  process.env.MCP_SERVER_URL ??
  "https://mcp-server-production-b54a.up.railway.app/mcp";

const GROQ_MCP_SERVER_URL =
  process.env.GROQ_MCP_SERVER_URL ?? MCP_SERVER_URL;

// Groq config
const GROQ_API_KEY = requireEnv("GROQ_API_KEY");
const GROQ_MODEL =
  process.env.GROQ_MODEL ?? "meta-llama/llama-4-scout-17b-16e-instruct";

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

// Groq Responses API, using OpenAI client with Groq baseURL
const groq = new OpenAI({
  apiKey: GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
});

// Simple chat history per conversationId, only chat-style messages:
// { role: "system" | "user" | "assistant", content: string }
type Role = "system" | "user" | "assistant";
type ChatMessage = { role: Role; content: string };

const conversations = new Map<string, ChatMessage[]>();

/**
 * POST /chat/openai
 * Body: { conversationId?: string; systemPrompt?: string; message: string }
 *
 * Uses Groq Responses API + remote MCP tools.
 * We keep the route name `/chat/openai` so the frontend doesn’t need to change.
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
      if (sys) {
        history.push({ role: "system", content: sys });
      }
    }

    // Add the new user message
    history.push({ role: "user", content: message });

    // MCP tool config for Groq (NO headers here — Groq rejected that)
    const mcpTool = {
      type: "mcp",
      server_label: "disabled-services",
      server_url: GROQ_MCP_SERVER_URL,
      allowed_tools: [...allowedTools],
      require_approval: "never",
    } as any;

    // Call Groq Responses API
    const response = await groq.responses.create({
      model: GROQ_MODEL,
      input: history,
      tools: [mcpTool],
      // Groq doesn’t support previous_response_id or store
    });

    const text = response.output_text ?? "";

    // Save assistant message into history for next turn
    if (text) {
      history.push({ role: "assistant", content: text });
    }
    conversations.set(cid, history);

    res.json({
      conversationId: cid,
      text,
    });
  } catch (err: any) {
    console.error("Groq /chat/openai error:", err);
    res.status(500).json({ error: err?.message || "Groq Responses error" });
  }
});

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

app.listen(PORT, () => {
  console.log(`AI Gateway (Groq) listening on http://localhost:${PORT}`);
  console.log(`GROQ_MODEL: ${GROQ_MODEL}`);
  console.log(`GROQ_MCP_SERVER_URL: ${GROQ_MCP_SERVER_URL}`);
});
