import { Outlet } from "react-router-dom";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";

export function RootLayout() {
    return (
        <SidebarProvider>
            <div className="flex min-h-screen w-full overflow-hidden">
                <AppSidebar />

                <div className="flex flex-col flex-1">
                <header className="flex h-12 items-center border-b px-4">
                    <SidebarTrigger />
                </header>

                <main className="flex-1 p-4">
                    <Outlet />
                </main>
                </div>
            </div>
        </SidebarProvider>
    );
}