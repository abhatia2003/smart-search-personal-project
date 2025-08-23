import { useState, type FormEvent, type ChangeEvent } from "react";
import "./../styles/popup.css";

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<string[]>([]);
  const [submitted, setSubmitted] = useState(false); 

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const dummyData = [
      "ArrayList in Java",
      "Arrays",
      "Arriving soon",
      "LinkedList in Java",
      "HashMap usage",
      "Binary Search Tree",
      "Graph traversal algorithms",
    ];

    const filtered = dummyData.filter((item) =>
      item.toLowerCase().includes(query.toLowerCase())
    );

    setResults(filtered);
    setSubmitted(true); 
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
    setSubmitted(false); 
  };

  return (
    <div className="popup-wrapper">
      <form onSubmit={handleSubmit} role="search">
        <label htmlFor="search">Search for stuff</label>
        <input
          id="search"
          type="search"
          placeholder="Search..."
          autoFocus
          required
          value={query}
          onChange={handleChange}
        />
        <button type="submit">Go</button>
      </form>
 
      {submitted && (
        <div className="results">
          {results.length > 0 ? (
            results.map((r, i) => (
              <div className="result-item" key={i}>
                {r}
              </div>
            ))
          ) : (
            <div className="no-results">No matches found</div>
          )}
        </div>
      )}
    </div>
  );
}