import type { FormEvent } from "react";
import "./../styles/popup.css";

export default function App() {
    const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        console.log("Search submitted!");
        // Access form values here if needed
      };
    
      return (
        <form onSubmit={handleSubmit} role="search">
          <label htmlFor="search">Search for stuff</label>
          <input
            id="search"
            type="search"
            placeholder="Search..."
            autoFocus
            required
          />
          <button type="submit">Go</button>
        </form>
      );
}