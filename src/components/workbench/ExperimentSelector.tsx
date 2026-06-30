import { useState, useEffect, useRef } from 'react';
import { apiService, type ExperimentData } from '../../services/api';

interface ExperimentSelectorProps {
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  label?: string;
  includeCurrent?: boolean;
  currentExperimentId?: string | null;
  className?: string;
}

export function ExperimentSelector({
  selectedId,
  onSelect,
  label = '选择实验',
  includeCurrent = true,
  currentExperimentId,
  className = '',
}: ExperimentSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [experiments, setExperiments] = useState<ExperimentData[]>([]);
  const [loading, setLoading] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const loadExperiments = async () => {
      setLoading(true);
      try {
        const res = await apiService.listExperiments({ page_size: 50 });
        if (res.data?.items) {
          const completed = res.data.items.filter(
            (e) => e.status === 'completed' || e.status === 'stopped'
          );
          setExperiments(completed);
        }
      } catch {
        // 静默失败
      } finally {
        setLoading(false);
      }
    };
    if (isOpen && experiments.length === 0) {
      loadExperiments();
    }
  }, [isOpen, experiments.length]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);

  const selectedExp = experiments.find((e) => e.experiment_id === selectedId);
  const isCurrent = currentExperimentId && selectedId === currentExperimentId;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-emerald-400';
      case 'running': return 'bg-blue-400 animate-pulse';
      case 'failed': return 'bg-red-400';
      case 'stopped': return 'bg-amber-400';
      default: return 'bg-slate-400';
    }
  };

  return (
    <div className={`relative ${className}`} ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 rounded-md border border-white/[0.08] bg-white/[0.03] px-2.5 py-1.5 text-[11px] transition-all hover:bg-white/[0.06] w-full"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary flex-shrink-0">
          <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
        </svg>
        <span className="truncate flex-1 text-left text-muted-foreground">
          {selectedExp ? (
            <span className="text-foreground/80">{selectedExp.name}</span>
          ) : isCurrent && currentExperimentId ? (
            <span className="text-emerald-400">当前训练实验</span>
          ) : (
            label
          )}
        </span>
        {selectedId && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onSelect(null);
            }}
            className="text-muted-foreground hover:text-red-400 flex-shrink-0"
            title="清除选择"
          >
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        )}
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={`text-muted-foreground transition-transform flex-shrink-0 ${isOpen ? 'rotate-180' : ''}`}>
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 right-0 z-50 mt-1 max-h-64 overflow-y-auto rounded-lg border border-white/[0.1] bg-[#161922] shadow-xl shadow-black/40">
          {loading ? (
            <div className="flex items-center justify-center py-4">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            </div>
          ) : experiments.length === 0 ? (
            <div className="px-3 py-4 text-center text-[11px] text-muted-foreground">
              暂无已完成的实验
            </div>
          ) : (
            <div className="py-1">
              {includeCurrent && currentExperimentId && (
                <button
                  onClick={() => {
                    onSelect(currentExperimentId);
                    setIsOpen(false);
                  }}
                  className={`flex w-full items-center gap-2 px-3 py-2 text-left text-[11px] transition-colors hover:bg-white/[0.06] ${
                    selectedId === currentExperimentId ? 'bg-emerald-500/10 text-emerald-400' : ''
                  }`}
                >
                  <span className="h-2 w-2 rounded-full bg-emerald-400" />
                  <span className="truncate flex-1">当前训练实验</span>
                  <span className="text-[9px] text-emerald-400/60">实时</span>
                </button>
              )}
              {experiments.map((exp) => (
                <button
                  key={exp.experiment_id}
                  onClick={() => {
                    onSelect(exp.experiment_id);
                    setIsOpen(false);
                  }}
                  className={`flex w-full items-center gap-2 px-3 py-2 text-left text-[11px] transition-colors hover:bg-white/[0.06] ${
                    selectedId === exp.experiment_id ? 'bg-primary/10 text-primary' : ''
                  }`}
                >
                  <span className={`h-2 w-2 rounded-full flex-shrink-0 ${getStatusColor(exp.status)}`} />
                  <span className="truncate flex-1">{exp.name}</span>
                  {exp.best_accuracy != null && (
                    <span className="text-[9px] text-muted-foreground font-mono flex-shrink-0">
                      {(exp.best_accuracy * 100).toFixed(1)}%
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
