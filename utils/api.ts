// utils/api.ts
// Helper for API calls to Flask backend

export async function predictAlzheimer(imageData: ArrayBuffer) {
  const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: Array.from(new Uint8Array(imageData)) })
  });
  return response.json();
}
