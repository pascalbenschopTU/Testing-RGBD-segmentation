using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;
using UnityEngine.Perception.Randomization.Utilities;

namespace SynthDet.Randomizers
{
    /// <summary>
    /// Creates multiple layers of evenly distributed but randomly placed objects
    /// </summary>
    [Serializable]
    [AddRandomizerMenu("SynthDet/Background Object Placement Randomizer")]
    public class BackgroundObjectPlacementRandomizer : Randomizer
    {
        [Tooltip("The Z offset component applied to all generated background layers.")]
        public FloatParameter depth = new FloatParameter { value = new UniformSampler(0.0f, 2.0f) };

        [Tooltip("The number of layers of objects to generate.")]
        public int layerCount = 1;

        [Tooltip("The scale factor to apply to the object's scale. The object's scale will be multiplied by a random value between the min and max values specified here.")]
		public FloatParameter scale = new FloatParameter { value = new UniformSampler(6f, 8f) };

        [Tooltip("The minimum distance between objects in the 2D plane.")]
        public float separationDistance = 0.6f;

        [Tooltip("The area in which to place objects in the 2D plane.")]
        public Vector2 placementArea = new Vector2(8f, 8f);

        [Tooltip("The prefabs to sample from for object generation.")]
        public CategoricalParameter<GameObject> prefabs;

        [Tooltip("If true, the color of the object will be adjusted.")]
        public bool adjustColor = false;

        [Tooltip("If true, calculate UVs for the object. This is useful for 3D objects that do not have UVs to apply textures.")]
        public bool calculateUVs = false;

        [Tooltip("If true, flatten the object along the depth axis.")]
        public bool flatten = false;

        [Tooltip("The materials to sample from for object color adjustment. (If Texture Randomizer is used, this will be overridden.)")]
        public CategoricalParameter<Material> materials;

        GameObject m_Container;
        GameObjectOneWayCache m_GameObjectOneWayCache;

        protected override void OnAwake()
        {
            if (calculateUVs)
            {
                Calculate3DUVW();
            }

            m_Container = new GameObject("Background Distractors");
            m_Container.transform.parent = scenario.transform;
            m_GameObjectOneWayCache = new GameObjectOneWayCache(m_Container.transform,
                prefabs.categories.Select((element) => element.Item1).ToArray(), this);
        }

        /// <summary>
        /// Generates background layers of objects at the start of each scenario iteration
        /// </summary>
        protected override void OnIterationStart()
        {
            for (var i = 0; i < layerCount; i++)
            {
                var placementSeed = SamplerState.NextRandomState();
                
                var placementSamples = PoissonDiskSampling.GenerateSamples(
                    placementArea.x, placementArea.y, separationDistance, placementSeed);
                var offset = new Vector3(placementArea.x, placementArea.y, 0f) * -0.5f;

                foreach (var sample in placementSamples) 
                {
                    var prefab = prefabs.Sample();
                    var scaleFactor = scale.Sample();
                    var depthOffset = depth.Sample();
                    var colorMaterial = materials.Sample();

                    var instance = m_GameObjectOneWayCache.GetOrInstantiate(prefab);
                    instance.transform.position = new Vector3(sample.x, sample.y, separationDistance * i + depthOffset) + offset;
                    instance.transform.localScale = new Vector3(scaleFactor, scaleFactor, scaleFactor);

                    if (flatten)
                    {
                        FlattenGameObject(instance);
                    }

                    if (adjustColor)
                    {
                        var renderer = instance.GetComponent<Renderer>();
                        if (renderer != null)
                        {
                            renderer.material = colorMaterial;
                        }
                    }
                }

                placementSamples.Dispose();
            }
        }
        
        /// <summary>
        /// Deletes generated background objects after each scenario iteration is complete
        /// </summary>
        protected override void OnIterationEnd()
        {
            m_GameObjectOneWayCache.ResetAllObjects();
        }

        private void Calculate3DUVW()
        {
            GameObject[] objects = prefabs.categories.Select((element) => element.Item1).ToArray();
            foreach (var obj in objects)
            {
                var meshRenderer = obj.GetComponent<MeshRenderer>();
                if (meshRenderer != null)
                {
                    var mesh = obj.GetComponent<MeshFilter>().sharedMesh;
                    var uv = mesh.uv;
                    var vertices = mesh.vertices;
                    var normals = mesh.normals;
                    var tangents = mesh.tangents;
                    // var uvs = new Vector3[vertices.Length];
                    var uvs = new Vector2[vertices.Length];
                    for (int i = 0; i < vertices.Length; i++)
                    {
                        var vertex = vertices[i];
                        var normal = normals[i];
                        var tangent = tangents[i];
                        var uvw = new Vector3(
                            Vector3.Dot(vertex, tangent),
                            Vector3.Dot(vertex, Vector3.Cross(normal, tangent)),
                            Vector3.Dot(vertex, normal)
                        );
                        uvs[i] = new Vector2(uvw.x, uvw.y);
                    }
                    mesh.uv = uvs;
                }
            }
        }

        private void FlattenGameObject(GameObject obj)
        {
            Mesh mesh = obj.GetComponent<MeshFilter>().mesh;
            Vector3[] vertices = mesh.vertices;

            // Find the average Y position of all vertices
            float averageY = 0f;
            foreach (Vector3 vertex in vertices)
            {
                averageY += vertex.y;
            }
            averageY /= vertices.Length;

            // Flatten vertices along the Y axis
            for (int i = 0; i < vertices.Length; i++)
            {
                vertices[i] = new Vector3(vertices[i].x, averageY, vertices[i].z);
            }

            // Update the mesh with the flattened vertices
            mesh.vertices = vertices;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
        }
    }
}