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
import { Link } from "react-router-dom";
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
                                <Link to="/" className="text-xs"><span>Patient Assesment</span></Link>
                            </SidebarMenuButton>
                        </SidebarMenuItem>
                        <SidebarMenuItem>
                            <SidebarMenuButton asChild tooltip="Inventory">
                                <Link to="/inventory" className="text-xs"><span>Medication Inventory</span></Link>
                            </SidebarMenuButton>
                        </SidebarMenuItem>

                        </SidebarMenu>
                </SidebarGroup>
                <SidebarGroup>
                    <SidebarGroupLabel>ML Resources</SidebarGroupLabel>
                        <SidebarMenu>
                        <SidebarMenuItem>
                            <SidebarMenuButton asChild tooltip="Home">
                                <Link to="/model_stats" className="text-xs"><span>ML Model Assesment</span></Link>
                            </SidebarMenuButton>
                        </SidebarMenuItem>
                        <SidebarMenuItem>
                            <SidebarMenuButton asChild tooltip="Scenarios">
                                <Link to="/scenarios" className="text-xs"><span>Scenario Comparison</span></Link>
                            </SidebarMenuButton>
                        </SidebarMenuItem>

                        </SidebarMenu>
                </SidebarGroup>
            </SidebarContent>


        </Sidebar>
    )
}
