First ever "Deep Research" prompt on NotebookLM:

Identify the primary technical challenges in fabricating scratch / specular holograms and propose practical implementation strategies using existing software pipelines.

Context:
I am developing a workflow that converts a 3D model (STL or heightmap) into scratch trajectories that can reproduce a specular hologram effect.

Constraints:

Prefer modifying existing slicers or CAM pipelines rather than building everything from scratch

Need both simulation rendering and theoretical validation

Focus on geometric optics (specular reflection) rather than wave interference holography

Output needed:

Mathematical models mapping view/light → scratch orientation

Existing software or research prototypes

Feasibility of modifying slicers (PrusaSlicer, Cura, etc.)

Known fabrication tolerances (scratch width, spacing, curvature resolution)

#2:
How can specular hologram or scratch hologram effects be modeled as a surface normal field encoding problem, and how can this be converted into manufacturable toolpaths using existing CAM or slicer software? Include known research, mathematical models, and examples of fabrication pipelines.

#3:
Given practical fabrication constraints (scratch width, spacing, tool radius, material reflectivity), how can a surface normal field solution be discretized into manufacturable scratch trajectories?

Timetable
Day 1
Run V1 → harvest:
papers
software names
keywords
fabrication constraints

Day 2
Run V2 → harvest:
math models
core equations
generalizable framework

Day 3
Run synthesis prompt → connect both worlds


⚠️ Hidden Meta Insight
Most people in this field are either:
👨‍🔬 optics theory only
OR
🛠 maker fabrication only
If you combine V1 + V2, you’re sitting in a rare middle zone, which is thesis gold and publication-worthy if done cleanly.