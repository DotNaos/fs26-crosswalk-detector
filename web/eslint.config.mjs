import tsParser from "@typescript-eslint/parser";

function createLineLimitRule(kind) {
  return {
    meta: {
      type: kind === "warning" ? "suggestion" : "problem",
      schema: [
        {
          type: "object",
          properties: {
            max: { type: "number" },
            label: { type: "string" },
          },
          additionalProperties: false,
        },
      ],
      messages: {
        tooManyLines: "{{label}}: {{count}} lines found, limit is {{max}}.",
      },
    },
    create(context) {
      const options = context.options[0] ?? {};
      const max = Number(options.max ?? 0);
      const label = String(options.label ?? "Line count");
      return {
        Program(node) {
          const count = context.sourceCode.lines.length;
          if (count <= max) return;
          context.report({
            node,
            messageId: "tooManyLines",
            data: { label, count: String(count), max: String(max) },
          });
        },
      };
    },
  };
}

const locPlugin = {
  rules: {
    "soft-max-lines": createLineLimitRule("warning"),
    "hard-max-lines": createLineLimitRule("error"),
  },
};

export default [
  {
    ignores: ["dist/**", "node_modules/**"],
  },
  {
    files: ["src/**/*.{ts,tsx}", "dev.ts", "vite.config.ts"],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "module",
        ecmaFeatures: {
          jsx: true,
        },
      },
    },
    plugins: {
      loc: locPlugin,
    },
    rules: {
      "loc/soft-max-lines": [
        "warn",
        {
          max: 500,
          label: "Soft file-size limit exceeded",
        },
      ],
      "loc/hard-max-lines": [
        "error",
        {
          max: 700,
          label: "Hard file-size limit exceeded",
        },
      ],
    },
  },
];
