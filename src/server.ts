// ai-gateway/src/server.ts
import express from "express";
import cors from "cors";
import OpenAI from "openai";
import {
  GoogleGenAI,
  FunctionCallingConfigMode,
  Type,
  type FunctionDeclaration,
} from "@google/genai";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v || v.trim() === "") throw new Error(`Missing env var: ${name}`);
  return v;
}

const PORT = Number(process.env.PORT ?? "3002");

const SYSTEM_PROMPsT = `You are an assistant for a POC that automates 3 backend actions via tools:
1) get_user_info(userId) -> returns {id, name, disabled}
2) create_parking_card(userId, userName) -> creates parking card (only if disabled=true)
3) create_vacation_request(userId, startDate, endDate, delegateUserId, delegateName)

Policy:
- Always call get_user_info before creating a parking card to confirm disabled=true.
- If user not disabled, do NOT create a parking card; explain why.
- For vacation requests: validate startDate <= endDate and dates are ISO yyyy-MM-dd.
- If any required data is missing, ask a short follow-up question.
- Confirm what you did after tool calls (summarize the action and return the created id).
- Never invent ids or tool results; only use tool outputs.
- Keep answers concise.`;


// IMPORTANT:
// - MCP_SERVER_URL can be an internal URL for the gateway->MCP client (Gemini path).
// - If you use OpenAI "hosted MCP tool", server_url must be publicly reachable by OpenAI.
//   So set OPENAI_MCP_SERVER_URL to a public https endpoint in real deployments / demos.
const MCP_SERVER_URL = process.env.MCP_SERVER_URL ?? "https://mcp-server-production-b54a.up.railway.app/mcp";
const OPENAI_MCP_SERVER_URL = process.env.OPENAI_MCP_SERVER_URL ?? MCP_SERVER_URL;

const OPENAI_API_KEY = requireEnv("OPENAI_API_KEY");
// const GEMINI_API_KEY = requireEnv("GEMINI_API_KEY");

const OPENAI_MODEL = process.env.OPENAI_MODEL ?? "gpt-5.1";
// const GEMINI_MODEL = process.env.GEMINI_MODEL ?? "gemini-1.5-pro";

const app = express();
app.use(cors());
app.use(express.json());

/** ---- MCP client (used for Gemini tool execution) ---- */
let mcpClient: Client | null = null;

async function getMcpClient(): Promise<Client> {
  if (mcpClient) return mcpClient;

  const client = new Client({ name: "ai-gateway-mcp-client", version: "1.0.0" });
  const transport = new StreamableHTTPClientTransport(new URL(MCP_SERVER_URL));
  await client.connect(transport);
  mcpClient = client;
  return client;
}

function extractTextFromMcpResult(result: any): string {
  if (result?.structuredContent) return JSON.stringify(result.structuredContent, null, 2);
  const firstText = result?.content?.find((c: any) => c.type === "text")?.text;
  if (typeof firstText === "string") return firstText;
  return JSON.stringify(result, null, 2);
}

export const allowedTools = [
  "get_user_info",
  "create_parking_card",
  "create_vacation_request",
  "request_house_maid",
  "request_home_checkup",

  // Medical Device Aid (POC)
  "list_assistive_devices",
  "submit_medical_device_aid_request",
  "get_medical_device_aid_request",
  "list_medical_device_aid_requests"

] as const;

export type AllowedToolName = (typeof allowedTools)[number];


/** ---- OpenAI endpoint: Responses API + hosted MCP tool ---- */
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// naive in-memory conversation state for POC
const openaiPrevResponseId = new Map<string, string>();

app.post("/chat/openai", async (req, res) => {
  try {
    const { conversationId, systemPrompt, message } = req.body as {
      conversationId?: string;
      systemPrompt?: string;
      message: string;
    };

    const cid = conversationId ?? "default";
    const prevId = openaiPrevResponseId.get(cid);

    const input = systemPrompt?.trim()
      ? [
        { role: "system" as const, content: systemPrompt.trim() },
        { role: "user" as const, content: message },
      ]
      : message;
    const response = await openai.responses.create({
      model: OPENAI_MODEL,
      input,
      previous_response_id: prevId,
      tools: [
        {
          type: "mcp",
          server_label: "disabled-services",
          server_url: OPENAI_MCP_SERVER_URL,
          allowed_tools: [...allowedTools],
          require_approval: "never",
        },
      ],
      store: true,
    });

    openaiPrevResponseId.set(cid, response.id);

    res.json({
      conversationId: cid,
      text: response.output_text,
      previous_response_id: response.id,
    });
  } catch (err: any) {
    console.error(err);
    res.status(500).json({ error: err?.message || "OpenAI error" });
  }
});

/** ---- Gemini endpoint: function calling + execute via MCP client ---- */
// const genai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });


const getUserInfoDecl: FunctionDeclaration = {
  name: "get_user_info",
  description: "Retrieve user info (including disabled flag) by userId.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      userId: { type: Type.STRING },
    },
    required: ["userId"],
  },
};

const createParkingCardDecl: FunctionDeclaration = {
  name: "create_parking_card",
  description: "Create a parking card for a disabled user (requires userId and userName).",
  parameters: {
    type: Type.OBJECT,
    properties: {
      userId: { type: Type.STRING },
      userName: { type: Type.STRING },
    },
    required: ["userId", "userName"],
  },
};

const createVacationDecl: FunctionDeclaration = {
  name: "create_vacation_request",
  description: "Create a vacation request (ISO dates yyyy-MM-dd + delegate userId & name).",
  parameters: {
    type: Type.OBJECT,
    properties: {
      userId: { type: Type.STRING },
      startDate: { type: Type.STRING },
      endDate: { type: Type.STRING },
      delegateUserId: { type: Type.STRING },
      delegateName: { type: Type.STRING },
    },
    required: ["userId", "startDate", "endDate", "delegateUserId", "delegateName"],
  },
};

// const geminiTools = [{ functionDeclarations: [getUserInfoDecl, createParkingCardDecl, createVacationDecl] }];

// app.post("/chat/gemini", async (req, res) => {
//   try {
//     const { message } = req.body as { message: string };

//     // 1) Ask Gemini (AUTO tool selection)
//     const first = await genai.models.generateContent({
//       model: GEMINI_MODEL,
//       contents: [{ role: "user", parts: [{ text: message }] }],
//       config: {
//         toolConfig: {
//           functionCallingConfig: {
//             mode: FunctionCallingConfigMode.AUTO, // ✅ enum
//           },
//         },
//         tools: geminiTools,
//       },
//     });

//     // In this SDK version, these are properties/getters (no parentheses).
//     const functionCalls = first.functionCalls ?? [];
//     if (functionCalls.length === 0) {
//       return res.json({ text: first.text ?? "" });
//     }

//     // 2) Execute tool calls via MCP
//     const client = await getMcpClient();

//     const modelParts = functionCalls.map((fc: any) => ({ functionCall: fc }));

//     const toolParts: any[] = [];

//     for (const fc of functionCalls as any[]) {
//       const toolName: string | undefined = fc?.name;
//       if (!toolName) {
//         toolParts.push({
//           functionResponse: {
//             name: "unknown_tool",
//             response: { error: "Missing tool name in function call" },
//           },
//         });
//         continue;
//       }

//       if (!allowedTools.includes(toolName as any)) {
//         toolParts.push({
//           functionResponse: {
//             name: toolName,
//             response: { error: "Tool not allowed" },
//           },
//         });
//         continue;
//       }

//       const mcpResult = await client.callTool({
//         name: toolName,
//         arguments: fc?.args ?? {},
//       });

//       toolParts.push({
//         functionResponse: {
//           name: toolName,
//           response: { result: extractTextFromMcpResult(mcpResult) },
//         },
//       });
//     }

//     // 3) Send tool responses back to Gemini to finalize answer
//     const final = await genai.models.generateContent({
//       model: GEMINI_MODEL,
//       contents: [
//         { role: "user", parts: [{ text: message }] },
//         { role: "model", parts: modelParts },
//         { role: "tool", parts: toolParts },
//       ],
//       config: { tools: geminiTools },
//     });

//     res.json({ text: final.text ?? "" });
//   } catch (err: any) {
//     console.error(err);
//     res.status(500).json({ error: err?.message || "Gemini error" });
//   }
// });

app.get("/health", (_, res) => res.json({ ok: true }));

app.listen(PORT, () => {
  console.log(`AI Gateway listening on http://localhost:${PORT}`);
  console.log(`MCP_SERVER_URL (gateway->mcp): ${MCP_SERVER_URL}`);
  console.log(`OPENAI_MCP_SERVER_URL (OpenAI->mcp): ${OPENAI_MCP_SERVER_URL}`);
});
