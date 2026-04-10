declare const Bun: {
  file(path: string): Blob & { size: number };
  spawn(command: string[], options?: { cwd?: string; stdout?: string; stderr?: string; stdin?: string }): {
    kill(): void;
    exited: Promise<number>;
  };
  serve(options: { port: number; fetch(request: Request): Response | Promise<Response> }): { port: number };
};

interface ImportMeta {
  readonly dir: string;
}
