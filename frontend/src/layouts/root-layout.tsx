import { Outlet } from "react-router-dom";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";

export function RootLayout() {
    return (
        <SidebarProvider>
            <div className="flex min-screen w-full overflow-hidden border-r-2">
                <AppSidebar />
                <main className="flex-1">
                    <SidebarTrigger />
                    <Outlet />
                </main>
            </div>
        </SidebarProvider>
    );
}