import { useState } from "react";
import "./../styles/popup.css";

export default function App() {
  const [query, setQuery] = useState("");

  return (
    <div className="popup-container">
      <h1>🔍 Smart Search</h1>
      <input
        type="text"
        placeholder="Search this page..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button>Search</button>
    </div>
  );
}