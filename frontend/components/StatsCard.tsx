interface StatsCardProps {
  label: string;
  value: string;
  sub?: string;
  pass?: boolean;
}

export default function StatsCard({ label, value, sub, pass }: StatsCardProps) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <p
        className={`text-2xl font-bold tabular-nums ${
          pass === true
            ? "text-emerald-400"
            : pass === false
            ? "text-red-400"
            : "text-white"
        }`}
      >
        {value}
      </p>
      {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
    </div>
  );
}
