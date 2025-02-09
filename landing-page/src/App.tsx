import {
  Route,
  createBrowserRouter,
  createRoutesFromElements,
  RouterProvider,
} from "react-router-dom";
import Hero from "@/Landing/Hero";

function App() {
  return (
    <RouterProvider
      router={createBrowserRouter(
        createRoutesFromElements(<Route index element={<Hero />} />),
      )}
    />
  );
}

export default App;
