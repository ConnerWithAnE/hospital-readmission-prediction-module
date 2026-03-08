import * as React from "react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { APITester } from "@/APITester";


export function AppSidebar() {
    return (
        <Sidebar>
            <SidebarHeader>
                <span className="font-bold text-lg">Risk Assessment</span>
            </SidebarHeader>
            <SidebarContent>
                
                <SidebarGroup>
                    <SidebarGroupLabel>Doctor Resources</SidebarGroupLabel>
                        <SidebarMenu>
                        <SidebarMenuItem>
                            <SidebarMenuButton asChild tooltip="Home">
                            <a href="/" className="text-xs"><span>Patient Assesment</span></a>
                            </SidebarMenuButton>
                        </SidebarMenuItem>

                        </SidebarMenu>
                </SidebarGroup>
            </SidebarContent>


        </Sidebar>
    )
}
