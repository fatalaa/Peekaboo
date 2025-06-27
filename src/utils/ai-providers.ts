import OpenAI from "openai";
import { GoogleGenAI } from "@google/genai";
import { Logger } from "pino";
import { AIProvider } from "../types/index.js";

export function parseAIProviders(aiProvidersEnv: string): AIProvider[] {
  if (!aiProvidersEnv || !aiProvidersEnv.trim()) {
    return [];
  }

  return aiProvidersEnv
    .split(/[,;]/) // Support both comma and semicolon separators
    .map((p) => p.trim())
    .filter(Boolean)
    .map((provider) => {
      const [providerName, model] = provider.split("/");
      return {
        provider: providerName?.trim() || "",
        model: model?.trim() || "",
      };
    })
    .filter((p) => p.provider && p.model);
}

export interface ProviderStatus {
  available: boolean;
  error?: string;
  details?: {
    modelAvailable?: boolean;
    serverReachable?: boolean;
    apiKeyPresent?: boolean;
    modelList?: string[];
  };
}

export async function isProviderAvailable(
  provider: AIProvider,
  logger: Logger,
): Promise<boolean> {
  const status = await getProviderStatus(provider, logger);
  return status.available;
}

export async function getProviderStatus(
  provider: AIProvider,
  logger: Logger,
): Promise<ProviderStatus> {
  try {
    switch (provider.provider.toLowerCase()) {
      case "ollama":
        return await checkOllamaStatus(provider.model, logger);
      case "openai":
        return await checkOpenAIStatus(provider.model, logger);
      case "anthropic":
        return checkAnthropicStatus(provider.model);
      case "gemini":
        return await checkGeminiStatus(provider.model, logger);
      default:
        logger.warn({ provider: provider.provider }, "Unknown AI provider");
        return {
          available: false,
          error: `Unknown provider: ${provider.provider}`,
        };
    }
  } catch (error) {
    logger.error(
      { error, provider: provider.provider },
      "Error checking provider status",
    );
    return {
      available: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

async function checkOllamaStatus(model: string, logger: Logger): Promise<ProviderStatus> {
  try {
    const baseUrl = process.env.PEEKABOO_OLLAMA_BASE_URL || "http://localhost:11434";

    // Check if server is reachable
    const tagsResponse = await fetch(`${baseUrl}/api/tags`, {
      signal: AbortSignal.timeout(3000), // 3 second timeout
    });

    if (!tagsResponse.ok) {
      return {
        available: false,
        error: `Ollama server returned ${tagsResponse.status}`,
        details: {
          serverReachable: false,
        },
      };
    }

    const tagsData = await tagsResponse.json();
    const availableModels = tagsData.models?.map((m: { name: string }) => m.name) || [];

    // Check if the specific model is available
    const modelAvailable = availableModels.some((m: string) =>
      m === model || m.startsWith(model + ":") || model.startsWith(m.split(":")[0]),
    );

    if (!modelAvailable) {
      return {
        available: false,
        error: `Model '${model}' not found. Available models: ${availableModels.join(", ") || "none"}`,
        details: {
          serverReachable: true,
          modelAvailable: false,
          modelList: availableModels,
        },
      };
    }

    return {
      available: true,
      details: {
        serverReachable: true,
        modelAvailable: true,
        modelList: availableModels,
      },
    };
  } catch (error) {
    logger.debug({ error }, "Ollama not available");
    const errorMessage = error instanceof Error ? error.message : "Unknown error";

    if (errorMessage.includes("fetch") || errorMessage.includes("timeout")) {
      return {
        available: false,
        error: "Ollama server not reachable (not running or network issue)",
        details: {
          serverReachable: false,
        },
      };
    }

    return {
      available: false,
      error: errorMessage,
      details: {
        serverReachable: false,
      },
    };
  }
}

async function checkOpenAIStatus(model: string, logger: Logger): Promise<ProviderStatus> {
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    return {
      available: false,
      error: "OpenAI API key not configured (OPENAI_API_KEY environment variable missing)",
      details: {
        apiKeyPresent: false,
      },
    };
  }

  try {
    // Test the API key by making a simple models list request
    const openai = new OpenAI({
      apiKey,
      timeout: 3000, // 3 second timeout
    });

    const modelsResponse = await openai.models.list();
    const availableModels = modelsResponse.data.map(m => m.id);

    // Check if the specific model is available
    const modelAvailable = availableModels.includes(model);

    if (!modelAvailable) {
      // For OpenAI, we'll be more lenient and just warn if model isn't in the list
      // since the models list API might not include all available models
      logger.debug({ model, availableCount: availableModels.length }, "Model not found in OpenAI models list, but this might be normal");
    }

    return {
      available: true,
      details: {
        apiKeyPresent: true,
        serverReachable: true,
        modelAvailable: modelAvailable,
        modelList: availableModels.slice(0, 10), // Limit to first 10 models for brevity
      },
    };
  } catch (error) {
    logger.debug({ error }, "OpenAI API check failed");
    const errorMessage = error instanceof Error ? error.message : "Unknown error";

    if (errorMessage.includes("401") || errorMessage.includes("Unauthorized")) {
      return {
        available: false,
        error: "Invalid OpenAI API key",
        details: {
          apiKeyPresent: true,
          serverReachable: true,
        },
      };
    }

    if (errorMessage.includes("network") || errorMessage.includes("fetch")) {
      return {
        available: false,
        error: "Cannot reach OpenAI API (network issue)",
        details: {
          apiKeyPresent: true,
          serverReachable: false,
        },
      };
    }

    return {
      available: false,
      error: `OpenAI API error: ${errorMessage}`,
      details: {
        apiKeyPresent: true,
        serverReachable: false,
      },
    };
  }
}

function checkAnthropicStatus(_model: string): ProviderStatus {
  const apiKey = process.env.ANTHROPIC_API_KEY;

  if (!apiKey) {
    return {
      available: false,
      error: "Anthropic API key not configured (ANTHROPIC_API_KEY environment variable missing)",
      details: {
        apiKeyPresent: false,
      },
    };
  }

  return {
    available: false,
    error: "Anthropic support not yet implemented",
    details: {
      apiKeyPresent: true,
    },
  };
}

async function checkGeminiStatus(model: string, logger: Logger): Promise<ProviderStatus> {
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    return {
      available: false,
      error: "Gemini API key not configured (GEMINI_API_KEY environment variable missing)",
      details: {
        apiKeyPresent: false,
      },
    };
  }

  try {
    // Note: The new @google/genai SDK doesn't provide a lightweight validation method
    // We'll just check if the API key is present
    return {
      available: true,
      details: {
        apiKeyPresent: true,
        serverReachable: true,
        modelAvailable: true,
      },
    };
  } catch (error) {
    logger.debug({ error }, "Gemini API check failed");
    const errorMessage = error instanceof Error ? error.message : "Unknown error";

    // Check for authentication errors
    if (
      errorMessage.includes("API key not valid") ||
      errorMessage.includes("401") ||
      errorMessage.includes("403") ||
      errorMessage.includes("Unauthorized") ||
      errorMessage.includes("API_KEY_INVALID")
    ) {
      return {
        available: false,
        error: "Invalid Google Gemini API key",
        details: {
          apiKeyPresent: true,
          serverReachable: true,
        },
      };
    }

    // Check for model availability errors
    if (
      errorMessage.includes("404") ||
      errorMessage.includes("Model not found") ||
      errorMessage.includes("models/") && errorMessage.includes("is not found")
    ) {
      return {
        available: false,
        error: `Model '${model}' not available. Try 'gemini-2.5-flash' or 'gemini-2.5-pro'.`,
        details: {
          apiKeyPresent: true,
          serverReachable: true,
          modelAvailable: false,
        },
      };
    }

    // Check for quota/rate limit errors
    if (
      errorMessage.includes("429") ||
      errorMessage.includes("quota") ||
      errorMessage.includes("rate limit")
    ) {
      return {
        available: false,
        error: "Gemini API quota exceeded or rate limited",
        details: {
          apiKeyPresent: true,
          serverReachable: true,
        },
      };
    }

    // Network errors
    if (
      errorMessage.includes("fetch") ||
      errorMessage.includes("network") ||
      errorMessage.includes("timeout") ||
      errorMessage.includes("ENOTFOUND") ||
      errorMessage.includes("ECONNREFUSED")
    ) {
      return {
        available: false,
        error: "Cannot reach Google Gemini API (network issue)",
        details: {
          apiKeyPresent: true,
          serverReachable: false,
        },
      };
    }

    return {
      available: false,
      error: `Gemini API error: ${errorMessage}`,
      details: {
        apiKeyPresent: true,
        serverReachable: false,
      },
    };
  }
}


export async function analyzeImageWithProvider(
  provider: AIProvider,
  imagePath: string,
  imageBase64: string,
  question: string,
  logger: Logger,
): Promise<string> {
  switch (provider.provider.toLowerCase()) {
    case "ollama":
      return await analyzeWithOllama(
        provider.model,
        imageBase64,
        question,
        logger,
      );
    case "openai":
      return await analyzeWithOpenAI(
        provider.model,
        imageBase64,
        question,
        logger,
      );
    case "anthropic":
      throw new Error("Anthropic support not yet implemented");
    case "gemini":
      return await analyzeWithGemini(
        provider.model,
        imageBase64,
        question,
        logger,
      );
    default:
      throw new Error(`Unsupported AI provider: ${provider.provider}`);
  }
}

async function analyzeWithOllama(
  model: string,
  imageBase64: string,
  question: string,
  logger: Logger,
): Promise<string> {
  const baseUrl =
    process.env.PEEKABOO_OLLAMA_BASE_URL || "http://localhost:11434";

  logger.debug({ model, baseUrl }, "Analyzing image with Ollama");

  // Default to describing the image if no question is provided
  const prompt = question.trim() || "Please describe what you see in this image.";

  const response = await fetch(`${baseUrl}/api/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      prompt,
      images: [imageBase64],
      stream: false,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    logger.error(
      { status: response.status, error: errorText },
      "Ollama API error",
    );
    throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
  }

  const result = await response.json();
  return result.response || "No response from Ollama";
}

async function analyzeWithOpenAI(
  model: string,
  imageBase64: string,
  question: string,
  logger: Logger,
): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OpenAI API key not configured");
  }

  logger.debug({ model }, "Analyzing image with OpenAI");

  const openai = new OpenAI({ apiKey });

  // Default to describing the image if no question is provided
  const prompt = question.trim() || "Please describe what you see in this image.";

  const response = await openai.chat.completions.create({
    model: model || "gpt-4o",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: prompt },
          {
            type: "image_url",
            image_url: {
              url: `data:image/jpeg;base64,${imageBase64}`,
            },
          },
        ],
      },
    ],
    max_tokens: 1000,
  });

  return response.choices[0]?.message?.content || "No response from OpenAI";
}

async function analyzeWithGemini(
  model: string,
  imageBase64: string,
  question: string,
  logger: Logger,
): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("Gemini API key not configured");
  }

  logger.debug({ model }, "Analyzing image with Gemini");

  const ai = new GoogleGenAI({ apiKey });

  // Default to describing the image if no question is provided
  const prompt = question.trim() || "Please describe what you see in this image.";

  try {
    // Convert base64 to data for Gemini
    const imagePart = {
      inlineData: {
        data: imageBase64,
        mimeType: "image/jpeg",
      },
    };

    const response = await ai.models.generateContent({
      model: model || "gemini-2.5-flash",
      contents: [
        {
          parts: [
            { text: prompt },
            imagePart,
          ],
        },
      ],
    });

    return response.text || "No response from Gemini";
  } catch (error) {
    logger.error({ error }, "Gemini API error during image analysis");
    throw new Error(`Gemini API error: ${error instanceof Error ? error.message : "Unknown error"}`);
  }
}

export function getDefaultModelForProvider(provider: string): string {
  switch (provider.toLowerCase()) {
    case "ollama":
      return "llava:latest";
    case "openai":
      return "gpt-4o";
    case "anthropic":
      return "claude-3-sonnet-20240229";
    case "gemini":
      return "gemini-2.5-flash";
    default:
      return "unknown";
  }
}

export async function determineProviderAndModel(
  providerConfig: { type?: string; model?: string } | undefined,
  configuredProviders: AIProvider[],
  logger: Logger,
): Promise<{ provider: string | null; model: string }> {
  const requestedProviderType = providerConfig?.type || "auto";
  const requestedModelName = providerConfig?.model;

  if (requestedProviderType !== "auto") {
    // Find specific provider in configuration
    const configuredProvider = configuredProviders.find(
      (p) => p.provider.toLowerCase() === requestedProviderType.toLowerCase(),
    );

    if (!configuredProvider) {
      throw new Error(
        `Provider '${requestedProviderType}' is not enabled in server's PEEKABOO_AI_PROVIDERS configuration.`,
      );
    }

    // Check if provider is available
    const available = await isProviderAvailable(configuredProvider, logger);
    if (!available) {
      throw new Error(
        `Provider '${requestedProviderType}' is configured but not currently available.`,
      );
    }

    const model =
      requestedModelName ||
      configuredProvider.model ||
      getDefaultModelForProvider(requestedProviderType);

    return {
      provider: requestedProviderType,
      model,
    };
  }

  // Auto mode - find first available provider
  for (const configuredProvider of configuredProviders) {
    const available = await isProviderAvailable(configuredProvider, logger);
    if (available) {
      const model =
        requestedModelName ||
        configuredProvider.model ||
        getDefaultModelForProvider(configuredProvider.provider);

      return {
        provider: configuredProvider.provider,
        model,
      };
    }
  }

  return { provider: null, model: "" };
}
