import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { HomePage } from './components/home/HomePage';
import { WorkbenchProvider } from './components/workbench/WorkbenchContext';
import { GlobalTrainingProvider } from './contexts/GlobalTrainingContext';
import { ToastProvider } from './contexts/ToastContext';
import { FloatingActions } from './components/FloatingActions';
import { ErrorBoundary } from './components/ErrorBoundary';
import { AppLayout } from './components/layout/AppLayout';

// 路由懒加载 — 按页面拆分 chunk，减小主包体积
const WorkbenchPage = lazy(() =>
  import('./pages/WorkbenchPage').then((m) => ({ default: m.WorkbenchPage }))
);
const VisualizationPage = lazy(() => import('./pages/VisualizationPage'));
const TrainingLogPage = lazy(() => import('./pages/TrainingLogPage'));
const AISettingsPage = lazy(() => import('./pages/AISettingsPage'));
const ExperimentsPage = lazy(() => import('./pages/ExperimentsPage'));
const ExperimentDetailPage = lazy(() => import('./pages/ExperimentDetailPage'));
const DatasetsPage = lazy(() => import('./pages/DatasetsPage'));
const AutoTrainingPage = lazy(() => import('./pages/AutoTrainingPage'));

// 懒加载页面统一 fallback
function PageFallback() {
  return (
    <div className="flex h-full w-full items-center justify-center bg-[#0d1018] text-slate-400">
      <div className="flex flex-col items-center gap-3">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-600 border-t-slate-300" />
        <span className="text-sm">加载中…</span>
      </div>
    </div>
  );
}

/** 工作台已有顶部播放控制，FloatingActions 在该路由下不渲染避免重复 */
function ConditionalFloatingActions() {
  const location = useLocation();
  if (location.pathname === '/workbench') return null;
  if (location.pathname === '/') return null;
  return <FloatingActions />;
}

function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <GlobalTrainingProvider>
          <WorkbenchProvider>
            <Suspense fallback={<PageFallback />}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route element={<AppLayout />}>
                <Route
                  path="/workbench"
                  element={
                    <ErrorBoundary>
                      <WorkbenchPage />
                    </ErrorBoundary>
                  }
                />
                <Route
                  path="/workbench/visualization"
                  element={
                    <ErrorBoundary>
                      <VisualizationPage />
                    </ErrorBoundary>
                  }
                />
                <Route
                  path="/workbench/training-log"
                  element={
                    <ErrorBoundary>
                      <TrainingLogPage />
                    </ErrorBoundary>
                  }
                />
                <Route
                  path="/workbench/ai-settings"
                  element={
                    <ErrorBoundary>
                      <AISettingsPage />
                    </ErrorBoundary>
                  }
                />
                <Route
                  path="/experiments"
                  element={
                    <ErrorBoundary>
                      <ExperimentsPage />
                    </ErrorBoundary>
                  }
                />
                <Route
                  path="/experiments/:id"
                  element={
                    <ErrorBoundary>
                      <ExperimentDetailPage />
                    </ErrorBoundary>
                  }
                />
                <Route
                  path="/datasets"
                  element={
                    <ErrorBoundary>
                      <DatasetsPage />
                    </ErrorBoundary>
                  }
                />
                <Route
                  path="/auto-training"
                  element={
                    <ErrorBoundary>
                      <AutoTrainingPage />
                    </ErrorBoundary>
                  }
                />
              </Route>
            </Routes>
            </Suspense>
            <ConditionalFloatingActions />
          </WorkbenchProvider>
        </GlobalTrainingProvider>
      </ToastProvider>
    </BrowserRouter>
  );
}

export default App;
