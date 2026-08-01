// import { useEffect, useState } from "react";
// import { useProfile } from "@/context/ProfileContext";
// import { fetchGradPrograms, GradProgram } from "@/hooks/use-grad-school";

// const GradPage = () => {
//   const { profile } = useProfile();
//   const [programs, setPrograms] = useState<GradProgram[]>([]);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState<string | null>(null);

//   const handleSearch = async () => {
//     setLoading(true);
//     setError(null);
//     try {
//       const data = await fetchGradPrograms(
//         profile.major,
//         profile.gpa,
//         profile.gre ?? 0,
//         profile.coursework ?? []
//       );
//       setPrograms(data.programs);
//     } catch (e) {
//       setError("Could not load programs. Is the backend running?");
//     } finally {
//       setLoading(false);
//     }
//   };

//   useEffect(() => {
//     if (profile.major) handleSearch();
//   }, [profile.major]);

//   return (
//     <div className="min-h-screen p-4 md:p-6 lg:p-8 max-w-4xl mx-auto">
//       <header className="mb-6">
//         <h1 className="text-2xl font-bold">Graduate School Matching</h1>
//         <p className="text-muted-foreground">
//           Based on: {profile.major} | GPA {profile.gpa} | GRE {profile.gre}
//         </p>
//       </header>  

//       <button onClick={handleSearch} disabled={loading}>
//         {loading ? "Searching..." : "Find Programs"}
//       </button>

//       {error && <p className="text-red-500 mt-4">{error}</p>}

//       <div className="mt-6 space-y-4">
//         {programs.map((p, i) => (
//           <div key={i} className="border rounded-lg p-4">
//             <h2 className="font-semibold">{p.university} — {p.program}</h2>
//             <p>Match: {Math.round(p.fit_score * 100)}%</p>
//             <p>GPA req: {p.requirements.gpa} | GRE req: {p.requirements.gre}</p>
//             <p>Deadline: {p.requirements.deadline}</p>
//             <details className="mt-2">
//               <summary className="cursor-pointer text-sm text-blue-500">
//                 View Generated SOP
//               </summary>
//               <p className="mt-2 text-sm">{p.sop}</p>
//             </details>  
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default GradPage;

import { useEffect, useState } from "react";
import { useProfile } from "@/context/ProfileContext";
import { useGradSchool, GradProgram } from "@/hooks/use-grad-school";
import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { motion } from "framer-motion";

const GradPage = () => {
  const { profile } = useProfile();
  const { searchPrograms, loading } = useGradSchool();
  const navigate = useNavigate();

  const [programs, setPrograms] = useState<GradProgram[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    setError(null);

    try {
      const data = await searchPrograms({
        major: profile.major,
        gpa: profile.gpa,
        gre: profile.gre ?? 0,
        coursework: profile.coursework ?? [],
      });

      setPrograms(data);
    } catch (e) {
      setError("Could not load programs. Is the backend running?");
    }
  };

  useEffect(() => {
    if (profile.major) handleSearch();
  }, [profile.major]);

  return (
    <div className="min-h-screen p-4 md:p-6 lg:p-8 max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center gap-3 mb-6"
      >
        <button
          onClick={() => navigate("/dashboard")}
          className="text-muted-foreground hover:text-neon transition-colors flex items-center"
          aria-label="Back to dashboard"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <span className="font-display text-sm font-semibold text-foreground">
          Dashboard
        </span>
      </motion.div>

      <header className="mb-6">
        <h1 className="text-2xl font-bold">Graduate School Matching</h1>
        <p className="text-muted-foreground">
          Based on: {profile.major} | GPA {profile.gpa} | GRE {profile.gre}
        </p>
      </header>

      <button onClick={handleSearch} disabled={loading}>
        {loading ? "Searching..." : "Find Programs"}
      </button>

      {error && <p className="text-red-500 mt-4">{error}</p>}

      <div className="mt-6 space-y-4">
        {programs.map((p, i) => (
          <div key={i} className="border rounded-lg p-4">
            <h2 className="font-semibold">
              {p.university} — {p.program}
            </h2>

            <p>Match: {Math.round(p.fit_score * 100)}%</p>

            <p>
              GPA req: {p.requirements.gpa} | GRE req:{" "}
              {p.requirements.gre}
            </p>

            <p>Deadline: {p.requirements.deadline}</p>

            <details className="mt-2">
              <summary className="cursor-pointer text-sm text-blue-500">
                View Generated SOP
              </summary>
              <p className="mt-2 text-sm">{p.sop}</p>
            </details>
          </div>
        ))}
      </div>
    </div>
  );
};

export default GradPage;