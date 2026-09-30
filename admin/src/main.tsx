import { renderAdminApp } from "@/app/render";

const root = document.getElementById("root");
if (!root) throw new Error("The page has no #root element.");
renderAdminApp(root);
