import {
  Route,
  createBrowserRouter,
  createRoutesFromElements,
  RouterProvider,
} from "react-router-dom";
import Hero from "@/Landing/Hero";
import MainApp from "./App/MainApp";

function App() {
  return (
    <RouterProvider
      router={createBrowserRouter(
        createRoutesFromElements(
          <Route path="/">
            <Route index element={<Hero />} />
            <Route path="/app" element={<MainApp />} />
          </Route>,
        ),
      )}
    />
  );
}

export default App;
