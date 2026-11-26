import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ACI — Agentic Credit Intelligence",
  description:
    "Plateforme bancaire d'analyse financière assistée par agents IA, routeur PyTorch/RL et contrôle humain.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
