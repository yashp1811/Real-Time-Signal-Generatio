import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Signal Generation Dashboard",
  description: "Real-Time Signal Generation & Backtesting Service",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full" style={{ background: "var(--bg-primary)", color: "var(--text-primary)" }}>
        {children}
      </body>
    </html>
  );
}
