import { Outlet } from 'react-router-dom';
import { SidebarNav } from './SidebarNav';

export function AppLayout() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0a0c14] text-white">
      <SidebarNav />
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
