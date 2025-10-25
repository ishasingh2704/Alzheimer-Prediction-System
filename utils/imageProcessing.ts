// utils/imageProcessing.ts
// Helper for image preprocessing (placeholder)

export function preprocessImage(file: File): Promise<Uint8Array> {
  // TODO: Implement actual preprocessing logic
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) {
        resolve(new Uint8Array(reader.result));
      }
    };
    reader.readAsArrayBuffer(file);
  });
}
