import { render } from "@solidjs/web";
import App from "./App.tsx";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("#root is missing from index.html");

render(() => <App />, root);
