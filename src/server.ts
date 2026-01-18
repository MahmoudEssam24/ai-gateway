// ai-gateway/src/server.ts
import express from "express";
import cors from "cors";
import OpenAI from "openai";
import Groq from "groq-sdk"; // Import Groq SDK
import {
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

const MCP_SERVER_URL = process.env.MCP_SERVER_URL ?? "https://mcp-server-production-b54a.up.railway.app/mcp";
const OPENAI_MCP_SERVER_URL = process.env.OPENAI_MCP_SERVER_URL ?? MCP_SERVER_URL;

// API Keys
const OPENAI_API_KEY = process.env.OPENAI_API_KEY; // Optional if only using Groq
const GROQ_API_KEY = requireEnv("GROQ_API_KEY");   // REQUIRED now

// Models
const OPENAI_MODEL = process.env.OPENAI_MODEL ?? "gpt-4o";
// Using the specific model you requested
const GROQ_MODEL = process.env.GROQ_MODEL ?? "meta-llama/llama-4-scout-17b-16e-instruct";

const app = express();
app.use(cors());
app.use(express.json());

/** ---- MCP client (Shared) ---- */
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

/** * ---- Tool Definitions (Groq/OpenAI Format) ---- 
 * We map your existing logic to standard JSON Schema for Groq 
 */
const tools: Groq.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_user_info",
      description: "Retrieve user info (including disabled flag) by userId.",
      parameters: {
        type: "object",
        properties: {
          userId: { type: "string" },
        },
        required: ["userId"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "create_parking_card",
      description: "Create a parking card for a disabled user.",
      parameters: {
        type: "object",
        properties: {
          userId: { type: "string" },
          userName: { type: "string" },
        },
        required: ["userId", "userName"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "create_vacation_request",
      description: "Create a vacation request (ISO dates yyyy-MM-dd).",
      parameters: {
        type: "object",
        properties: {
          userId: { type: "string" },
          startDate: { type: "string" },
          endDate: { type: "string" },
          delegateUserId: { type: "string" },
          delegateName: { type: "string" },
        },
        required: ["userId", "startDate", "endDate", "delegateUserId", "delegateName"],
      },
    },
  },
  // Add other tools here (house_maid, etc) as needed
];


/** ---- Groq Endpoint ---- */
const groq = new Groq({ apiKey: GROQ_API_KEY });

app.post("/chat/openai", async (req, res) => {
  try {
    const { message, systemPrompt } = req.body;

    // 1. Build initial messages
    const messages: Groq.Chat.ChatCompletionMessageParam[] = [];
    if (systemPrompt) {
      messages.push({ role: "system", content: systemPrompt });
    }
    messages.push({ role: "user", content: message });

    // 2. First call to Groq (ask for intent/tools)
    // Note: We do NOT stream here to simplify the tool-loop logic
    const runner = await groq.chat.completions.create({
      model: GROQ_MODEL,
      messages: messages,
      tools: tools,
      tool_choice: "auto",
      temperature: 1,
      max_completion_tokens: 1024,
      top_p: 1,
    });

    const responseMessage = runner.choices[0].message;

    // 3. Check if Groq wants to run tools
    if (responseMessage.tool_calls && responseMessage.tool_calls.length > 0) {
      // Add the model's request to history
      messages.push(responseMessage);

      const client = await getMcpClient();

      // 4. Loop through tool calls and execute via MCP
      for (const toolCall of responseMessage.tool_calls) {
        const functionName = toolCall.function.name;
        const functionArgs = JSON.parse(toolCall.function.arguments);

        console.log(`[Groq] Calling MCP tool: ${functionName}`);

        try {
          // Execute via MCP Client
          const mcpResult = await client.callTool({
            name: functionName,
            arguments: functionArgs,
          });

          const toolOutput = extractTextFromMcpResult(mcpResult);

          // Append tool result to history
          messages.push({
            tool_call_id: toolCall.id,
            role: "tool",
            content: toolOutput,
          });
        } catch (error: any) {
          console.error(`Error executing ${functionName}:`, error);
          messages.push({
            tool_call_id: toolCall.id,
            role: "tool",
            content: JSON.stringify({ error: error.message }),
          });
        }
      }

      // 5. Second call to Groq (with tool results)
      const finalResponse = await groq.chat.completions.create({
        model: GROQ_MODEL,
        messages: messages,
      });

      return res.json({
        text: finalResponse.choices[0].message.content,
      });
    }

    // No tools called, just return text
    res.json({ text: responseMessage.content });

  } catch (err: any) {
    console.error("Groq Error:", err);
    res.status(500).json({ error: err?.message || "Groq error" });
  }
});


/** ---- Existing OpenAI Endpoint (Optional / Fallback) ---- */
// You can keep this if you still want to support OpenAI
// if (OPENAI_API_KEY) {
//   const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
//   const openaiPrevResponseId = new Map<string, string>();

//   app.post("/chat/openai", async (req, res) => {
//     try {
//       const { conversationId, systemPrompt, message } = req.body;
//       const cid = conversationId ?? "default";
//       const prevId = openaiPrevResponseId.get(cid);

//       const input = systemPrompt?.trim()
//         ? [
//           { role: "system" as const, content: systemPrompt.trim() },
//           { role: "user" as const, content: message },
//         ]
//         : message;

//       const response = await openai.responses.create({
//         model: OPENAI_MODEL,
//         input,
//         previous_response_id: prevId,
//         tools: [
//           {
//             type: "mcp", // OpenAI Hosted MCP
//             server_label: "disabled-services",
//             server_url: OPENAI_MCP_SERVER_URL,
//             allowed_tools: ["get_user_info", "create_parking_card", "create_vacation_request"], 
//             require_approval: "never", 
//           },
//         ],
//         store: true,
//       });
//       openaiPrevResponseId.set(cid, response.id);
//       res.json({ conversationId: cid, text: response.output_text });
//     } catch (err: any) {
//       console.error(err);
//       res.status(500).json({ error: err.message });
//     }
//   });
// }

app.get("/health", (_, res) => res.json({ ok: true }));

app.listen(PORT, () => {
  console.log(`AI Gateway listening on http://localhost:${PORT}`);
  console.log(`Groq Model: ${GROQ_MODEL}`);
});