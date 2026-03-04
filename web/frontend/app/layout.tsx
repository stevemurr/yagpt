import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'YAGPT — Pipeline Control',
  description: 'Node-graph web interface for the YAGPT LLM pipeline',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
