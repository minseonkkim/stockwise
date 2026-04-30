export default function SignalBadge({ signal }: { signal: "buy" | "hold" }) {
  return signal === "buy" ? (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/30">
      BUY
    </span>
  ) : (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-slate-700 text-slate-400">
      HOLD
    </span>
  );
}
