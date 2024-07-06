using System;
using System.Linq;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Perception.Randomization;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;
using UnityEngine.Perception.Randomization.Utilities;

namespace SynthDet.Randomizers
{
    /// <summary>
    /// Creates a 2D layer of evenly spaced GameObjects from a given list of prefabs
    /// </summary>
    [Serializable]
    [AddRandomizerMenu("SynthDet/Foreground Object Placement Randomizer")]
    public class ForegroundObjectPlacementRandomizer : Randomizer
    {
        [Tooltip("The maximum number of objects that will be placed in the scene.")]
        public int maxObjectCount = 15;

        [Tooltip("Enable this to randomize the number of objects that will be placed in the scene. The number of objects will be between 1 and the maximum object count.")]
        public bool randomizeObjectCount = false;
        
        [Tooltip("The assets to be placed in the scene.")]
        public AssetSource<GameObject> foregroundAssets;
        GameObject[] m_ProcessedAssetInstances;
        IntegerParameter m_ObjectIndexParameter = new IntegerParameter();

        [Tooltip("The depth of the objects in the scene. The objects will be placed at a random depth between the near and far clip planes of the camera.")]
        public FloatParameter depth = new FloatParameter { value = new UniformSampler(3.0f, 4.0f) };

        [Tooltip("The distance between objects in the scene. (in 2D - excluding depth differenes)")]
        public float separationDistance = 1.2f;
        [Tooltip("The area in which the objects will be placed in the scene. (in 2D - excluding depth differenes)")]
        public Vector2 placementArea = new Vector2(4f, 4f);

        GameObject m_Container;
        GameObjectOneWayCache m_GameObjectOneWayCache;

        [Tooltip("Enable this to normalize mesh sizes across all included objects, so that they all have a similar size before any further scale adjustments are applied during randomization. Note that this flag greatly influences the size of the objects on the screen, so any scale randomization will need to be adjusted based on the state of this flag.")]
        public bool normalizeObjectBounds = true;

        [Tooltip("The pre-scale factor to be applied to the objects. This is applied after the object bounds are normalized, but before the random scale factor is applied.")]
		public float preScale = 1.0f;

        [Tooltip("Enable this to remove the texture from the objects.")]
        public bool textureLess = false;

        [Tooltip("The material to be applied to the objects when the texture is removed.")]
        public Material textureLessMaterial;

        [Tooltip("Enter the range of hue values to be applied to the objects. Each object will get a unique hue value within this range. The minimum value is -180 and the maximum value is 180.")]
        public Vector2 hueRange = new Vector2(-180, 180);

        [Tooltip("Enable this to flatten the objects along the depth axis.")]
        public bool flatten = false;

        [Tooltip("Enable this to calculate the UVs of the objects, if the objects do not have UVs already (for applying textures). This will only work for objects that have a MeshFilter component.")]
        public bool calculateUVs = false;

        [Tooltip("Enable this to sample the objects evenly. The first two objects will be sampled from the least used objects.")]
        public bool sampleObjectsEvenly = true;

        private int iteration_count;

        private Dictionary<int, int> objectCounts = new Dictionary<int, int>();

        static readonly int k_HueOffsetShaderProperty = Shader.PropertyToID("_HueOffset");
        
        protected override void OnScenarioStart()
        {
            var constants = scenario.GetType().GetField("constants").GetValue(scenario);
            iteration_count = (int)constants.GetType().GetField("iterationCount").GetValue(constants);

            m_Container = new GameObject("Foreground Objects");
            var transform = scenario.transform;
            m_Container.transform.parent = transform;
            m_ProcessedAssetInstances = foregroundAssets.CreateProcessedInstances();
            
            m_ProcessedAssetInstances = m_ProcessedAssetInstances.Where(p =>
            {
                var isValid = ComputeBoundsUnchecked(p).IsValid;
                if (!isValid)
                    Debug.LogError($"Object {p} does not contain a mesh");

                return isValid;
            }).ToArray();

            for (int i = 0; i < m_ProcessedAssetInstances.Length; i++)
            {
                objectCounts[i] = 0;
            }

            if (calculateUVs)
            {
                Calculate3DUVW();
            }

            if (textureLess)
            {
                for (int i = 0; i < m_ProcessedAssetInstances.Length; i++)
                {
                    AssignTextureLessMaterial(m_ProcessedAssetInstances[i], i);
                }
            }
            
            m_GameObjectOneWayCache = new GameObjectOneWayCache(m_Container.transform, m_ProcessedAssetInstances, this);
            m_ObjectIndexParameter.value = new UniformSampler(0, m_ProcessedAssetInstances.Length);
        }
    
        /// <summary>
        /// Generates a foreground layer of objects at the start of each scenario iteration
        /// </summary>
        protected override void OnIterationStart()
        {
            PlaceObjects();
        }

        void PlaceObjects()
        {
            var spawnedCount = 0;
            var seed = SamplerState.NextRandomState();
            UnityEngine.Random.InitState((int)seed);
            var placementSamples = PoissonDiskSampling.GenerateSamples(
                placementArea.x, placementArea.y, separationDistance, seed);
            var offset = new Vector3(placementArea.x, placementArea.y, 0) * -0.5f;


            int tempMaxObjectCount = maxObjectCount;
            if (randomizeObjectCount && maxObjectCount > 1)
            {
                tempMaxObjectCount = UnityEngine.Random.Range(1, maxObjectCount);
            }
        
            foreach (var sample in placementSamples)
            {
                var index = Math.Min(m_ProcessedAssetInstances.Length, m_ObjectIndexParameter.Sample());
                if (sampleObjectsEvenly && spawnedCount < 2)
                {
                    index = SelectPrefabIndex(objectCounts);;
                }
                var prefab = m_ProcessedAssetInstances[index];
                objectCounts[index] += 1;
                var instance = m_GameObjectOneWayCache.GetOrInstantiate(prefab);

                if (flatten)
                {
                    FlattenGameObject(instance);
                }

                if (normalizeObjectBounds)
                {
                    instance.transform.localPosition = Vector3.zero;
                    instance.transform.localScale = Vector3.one;
                    instance.transform.localRotation = Quaternion.identity;
                    var bounds = ComputeBounds(instance);

                    Camera camera = Camera.main;
                    Vector3 nearA = camera.ViewportToWorldPoint(new Vector3(1, 0, camera.nearClipPlane));
                    Vector3 nearB = camera.ViewportToWorldPoint(new Vector3(0, 1, camera.nearClipPlane));

                    Vector3 farA = camera.ViewportToWorldPoint(new Vector3(1, 0, camera.farClipPlane));
                    Vector3 farB = camera.ViewportToWorldPoint(new Vector3(0, 1, camera.farClipPlane));

                    float depth_coordinate = depth.Sample();
                    float depth_scale = (depth_coordinate - camera.farClipPlane) / (camera.nearClipPlane - camera.farClipPlane);
                    float minXCoordinates = nearA.x + (farA.x - nearA.x) * depth_scale;
                    float maxXCoordinates = nearB.x + (farB.x - nearB.x) * depth_scale;
                    float minYCoordinates = nearA.y + (farA.y - nearA.y) * depth_scale;
                    float maxYCoordinates = nearB.y + (farB.y - nearB.y) * depth_scale;

                    Vector3 newOffset = offset - bounds.center;

                    float XCoordinate = Math.Min(Math.Max(minXCoordinates, sample.x + newOffset.x), maxXCoordinates);
                    float YCoordinate = Math.Min(Math.Max(minYCoordinates, sample.y + newOffset.y), maxYCoordinates);

                    instance.transform.localPosition = new Vector3(XCoordinate, YCoordinate, depth_coordinate);
                    var scale = instance.transform.localScale;
                    var magnitude = bounds.extents.magnitude;
                    scale.Scale(new Vector3(1/magnitude, 1/magnitude, 1/magnitude));
                    instance.transform.localScale = scale;
                }
                else
                {
                    instance.transform.position = new Vector3(sample.x, sample.y, depth.Sample()) + offset;    
                }

				instance.transform.localScale *= preScale;
                
                if (++spawnedCount == tempMaxObjectCount)
                    break;
            }
            placementSamples.Dispose();
        }

        /// <summary>
        /// Deletes generated foreground objects after each scenario iteration is complete
        /// </summary>
        protected override void OnIterationEnd()
        {
            m_GameObjectOneWayCache.ResetAllObjects();
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

        static NativeArray<Bounds> ComputeObjectBounds(GameObject[] prefabs)
        {
            var objectBounds = new NativeArray<Bounds>(prefabs.Length, Allocator.TempJob);
            for (int i = 0; i < prefabs.Length; i++)
            {
                var bounds = ComputeBounds(prefabs[i]);
                //assume objects will be aligned at origin
                bounds.center = Vector3.zero;
                objectBounds[i] = bounds;
            }

            return objectBounds;
        }
        
        public static Bounds ComputeBounds(GameObject gameObject)
        {
            var bounds = ComputeBoundsUnchecked(gameObject);
            if (!bounds.IsValid)
                throw new ArgumentException($"GameObject {gameObject.name} must have a MeshFilter in its hierarchy.");

            var result = new Bounds();
            result.SetMinMax(bounds.Min, bounds.Max);
            return result;
        }

        static SynthDetMinMaxAABB ComputeBoundsUnchecked(GameObject gameObject)
        {
            SynthDetMinMaxAABB aabb = new SynthDetMinMaxAABB(
                new float3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity), 
                new float3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity));
            var meshFilter = gameObject.GetComponent<MeshFilter>();
            if (meshFilter != null)
            {
                var bounds = meshFilter.sharedMesh.bounds;
                aabb = SynthDetMinMaxAABB.CreateFromCenterAndExtents(bounds.center, bounds.extents);
            }

            var transform = gameObject.transform;
            for (int i = 0; i < transform.childCount; i++)
            {
                var childAabb = ComputeBoundsUnchecked(transform.GetChild(i).gameObject);
                aabb.Encapsulate(childAabb);
            }

            aabb = SynthDetMinMaxAABB.Transform(float4x4.TRS(transform.localPosition, transform.localRotation, transform.localScale), aabb);
            return aabb;
        }

        private void AssignTextureLessMaterial(GameObject obj, int index)
        {
            var meshRenderer = obj.GetComponent<MeshRenderer>();
            if (meshRenderer != null) {
                meshRenderer.material = textureLessMaterial;
                meshRenderer.material.SetFloat(k_HueOffsetShaderProperty, GetObjectSpecificHue(index));
            } else {
                foreach (Transform child in obj.transform)
                {
                    AssignTextureLessMaterial(child.gameObject, index);
                }
            }
        }
        
        private int GetObjectSpecificHue(int index) {
            if (index < 0 || index >= m_ProcessedAssetInstances.Length)
            {
                Debug.LogError("The index must be between 0 and the number of objects.");
                return 0;
            }
            if (hueRange.x > hueRange.y)
            {
                Debug.LogError("The minimum value of the hue range must be less than the maximum value.");
                return 0;
            }
            if (hueRange.x < -180 || hueRange.y > 180)
            {
                Debug.LogError("The hue range must be between -180 and 180.");
                return 0;
            }

            int objectsCount = m_ProcessedAssetInstances.Length;
            double hueDouble = (((double)index / objectsCount) - 0.5) * (hueRange.y - hueRange.x);
            int hue = (int)Math.Clamp(hueDouble, hueRange.x, hueRange.y);
            
            return hue;
        }

        private void Calculate3DUVW()
        {
            foreach (var obj in m_ProcessedAssetInstances)
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

        private int SelectPrefabIndex(Dictionary<int, int> objectCounts)
        {
            // Find the minimum count
            int minCount = objectCounts.Values.Min();

            // Get a list of indices with the minimum count
            List<int> leastUsedIndices = objectCounts
                .Where(pair => pair.Value == minCount)
                .Select(pair => pair.Key)
                .ToList();

            // Select a random index from the least used indices
            int selectedIndex = leastUsedIndices[UnityEngine.Random.Range(0, leastUsedIndices.Count)];

            return selectedIndex;
        }
    }
}

