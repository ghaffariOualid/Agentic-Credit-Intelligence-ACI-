"use client";

import React from "react";
import { Cpu, Wifi, WifiOff } from "lucide-react";

interface TapisHeroBannerProps {
  sectionLabel: string;
  title: string;
  subtitle: string;
  apiOnline: boolean;
  vllmConfigured: boolean;
}

export function TapisHeroBanner({
  sectionLabel,
  title,
  subtitle,
  apiOnline,
  vllmConfigured,
}: TapisHeroBannerProps) {
  return (
    <header className="tapis-banner rounded-2xl overflow-hidden mb-6 p-4 md:p-5 flex items-center justify-center">
      {/* Cartouche central blanc verre dépoli : contraste parfait + motifs tapis intacts sur les flancs */}
      <div className="bg-white/92 backdrop-blur-md border border-white/90 shadow-sm rounded-xl px-7 py-4 max-w-2xl w-full text-center">
        <div className="flex flex-wrap items-center justify-center gap-2 mb-1.5">
          <span className="text-[11px] font-semibold tracking-widest uppercase text-[#9A4424]">
            {sectionLabel}
          </span>
          <span className="text-slate-300">•</span>
          <span
            className={`inline-flex items-center gap-1 text-[11px] font-medium ${
              apiOnline ? "text-emerald-700" : "text-slate-500"
            }`}
          >
            {apiOnline ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {apiOnline ? "API connectée" : "API hors ligne"}
          </span>
          {apiOnline && (
            <>
              <span className="text-slate-300">•</span>
              <span className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-600">
                <Cpu className="w-3 h-3 text-slate-500" />
                {vllmConfigured ? "Colab T4 vLLM" : "Fallback local"}
              </span>
            </>
          )}
        </div>

        <h1 className="text-xl md:text-2xl font-semibold tracking-tight text-slate-900">
          {title}
        </h1>
        <p className="mt-1 text-xs md:text-sm text-slate-500">{subtitle}</p>
      </div>
    </header>
  );
}
