using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;
using CameraRandomizerTag = SynthDet.RandomizerTags.CameraRandomizerTag;

/// <summary>
/// Creates a 2D layer of of evenly spaced GameObjects from a given list of prefabs
/// </summary>
[Serializable]
[AddRandomizerMenu("Wall Placement Randomizer")]
public class WallPlacementRandomizer : Randomizer
{
    /// <summary>
    /// The Samplers used to place objects in 3D space. Defaults to a uniform distribution in x, normal distribution
    /// in y, and constant value in z. These Samplers can be modified from the Inspector or via code.
    /// </summary>
    public Vector3Parameter positionDistribution = new()
    {
        x = new ConstantSampler(0f),
        y = new ConstantSampler(0f),
        z = new ConstantSampler(0f)
    };

    public Vector3Parameter scaleDistribution = new()
    {
        x = new ConstantSampler(100f),
        y = new ConstantSampler(1f),
        z = new ConstantSampler(100f)
    };

    public bool adjustColor = false;

	/// <summary>
	/// The sampler controlling the rgb color chosen for the background wall.
	/// </summary>
    public CategoricalParameter<Material> materials;

    /// <summary>
    /// The list of Prefabs to choose from
    /// </summary>
    public CategoricalParameter<GameObject> prefabs;

    public bool addSideWalls = false;

    [Tooltip("The size of the side walls in Unity units")]
    public float sideWallSize = 50f;

    public float sideWallWidth = 1f;

    [Tooltip("The ratio of the offset from the near clip plane to the far clip plane, 0.1 is 10% of the way from the near clip plane to the far clip plane")]
    public float positionNearFarRatio = 0.8f;

    [Tooltip("The angle offset from the camera's field of view for the side walls in degrees, 0 means the side walls are perpendicular to the camera's view frustum")]
    public float angleOffset = 0.5f;

    public float positionOffset = 0.5f;

    //The container object that will be the parent of all placed objects from this Randomizer
    GameObject m_Container;
    //This cache allows objects to be reused across placements
    UnityEngine.Perception.Randomization.Utilities.GameObjectOneWayCache m_GameObjectOneWayCache;

    /// <inheritdoc/>
    protected override void OnAwake()
    {
        m_Container = new GameObject("Objects");
        m_Container.transform.parent = scenario.transform;
        m_GameObjectOneWayCache = new UnityEngine.Perception.Randomization.Utilities.GameObjectOneWayCache(
            m_Container.transform, prefabs.categories.Select(element => element.Item1).ToArray(), this);
    }

    /// <summary>
    /// Generates a foreground layer of objects at the start of each Scenario Iteration
    /// </summary>
    protected override void OnIterationStart()
    {
        var seed = SamplerState.NextRandomState();
        UnityEngine.Random.InitState((int)seed);
        
        var instance = m_GameObjectOneWayCache.GetOrInstantiate(prefabs.Sample());
        instance.transform.position = positionDistribution.Sample();
        if (adjustColor)
        {
            instance.GetComponent<Renderer>().material = materials.Sample();
        }

        instance.transform.localScale = scaleDistribution.Sample();

        if (addSideWalls)
        {
            var cameraTags = tagManager.Query<CameraRandomizerTag>();
            foreach (var tag in cameraTags)
            {
                var camera = tag.GetComponent<Camera>();
                var cameraPosition = camera.transform.position;
                var cameraForward = camera.transform.forward;
                var cameraRight = camera.transform.right;
                var cameraUp = camera.transform.up;
                var FoV = camera.fieldOfView;

                var nearClipPlane = camera.nearClipPlane;
                var farClipPlane = camera.farClipPlane;

                // Calculate corners of near clip plane
                Vector3 nearCenter = cameraPosition + cameraForward * nearClipPlane;
                float nearHeight = 2.0f * Mathf.Tan(camera.fieldOfView * 0.5f * Mathf.Deg2Rad) * nearClipPlane;
                float nearWidth = nearHeight * camera.aspect;

                Vector3 A = nearCenter - (cameraRight * nearWidth / 2) - (cameraUp * nearHeight / 2);
                Vector3 B = nearCenter + (cameraRight * nearWidth / 2) - (cameraUp * nearHeight / 2);
                Vector3 C = nearCenter + (cameraRight * nearWidth / 2) + (cameraUp * nearHeight / 2);
                Vector3 D = nearCenter - (cameraRight * nearWidth / 2) + (cameraUp * nearHeight / 2);

                // Calculate corners of far clip plane
                Vector3 farCenter = cameraPosition + cameraForward * farClipPlane;
                float farHeight = 2.0f * Mathf.Tan(camera.fieldOfView * 0.5f * Mathf.Deg2Rad) * farClipPlane;
                float farWidth = farHeight * camera.aspect;

                Vector3 E = farCenter - (cameraRight * farWidth / 2) - (cameraUp * farHeight / 2);
                Vector3 F = farCenter + (cameraRight * farWidth / 2) - (cameraUp * farHeight / 2);
                Vector3 G = farCenter + (cameraRight * farWidth / 2) + (cameraUp * farHeight / 2);
                Vector3 H = farCenter - (cameraRight * farWidth / 2) + (cameraUp * farHeight / 2);
                
                // Function to calculate offset position
                Vector3 CalculateOffsetPosition(Vector3 nearPoint, Vector3 farPoint)
                {
                    return nearPoint + (farPoint - nearPoint) * positionNearFarRatio;
                }

                // Create wall 1 (bottom wall between A-B and E-F)
                var wall1 = m_GameObjectOneWayCache.GetOrInstantiate(prefabs.Sample());
                Vector3 centerABEF = CalculateOffsetPosition((A + B) / 2, (E + F) / 2);
                wall1.transform.position = centerABEF + Vector3.down * positionOffset;
                wall1.transform.rotation = Quaternion.Euler(-FoV / 2 + angleOffset, 0, 0);
                wall1.transform.localScale = new Vector3(100, sideWallWidth, sideWallSize);
                Renderer renderer1 = wall1.GetComponent<Renderer>();
                renderer1.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                if (adjustColor)
                {
                    wall1.GetComponent<Renderer>().material = materials.Sample();
                }

                // Create wall 2 (right wall between B-C and F-G)
                var wall2 = m_GameObjectOneWayCache.GetOrInstantiate(prefabs.Sample());
                Vector3 centerBCFG = CalculateOffsetPosition((B + C) / 2, (F + G) / 2);
                wall2.transform.position = centerBCFG + Vector3.left * positionOffset;
                wall2.transform.rotation = Quaternion.Euler(0, 90 + FoV / 2 - angleOffset, 0);
                wall2.transform.localScale = new Vector3(sideWallSize, 100, sideWallWidth);
                Renderer renderer2 = wall2.GetComponent<Renderer>();
                renderer2.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                if (adjustColor)
                {
                    wall2.GetComponent<Renderer>().material = materials.Sample();
                }


                // Create wall 3 (top wall between C-D and G-H)
                var wall3 = m_GameObjectOneWayCache.GetOrInstantiate(prefabs.Sample());
                Vector3 centerCDGH = CalculateOffsetPosition((C + D) / 2, (G + H) / 2);
                wall3.transform.position = centerCDGH + Vector3.up * positionOffset;
                wall3.transform.rotation = Quaternion.Euler(FoV / 2 - angleOffset, 0, 0);
                wall3.transform.localScale = new Vector3(100, sideWallWidth, sideWallSize);
                Renderer renderer3 = wall3.GetComponent<Renderer>();
                renderer3.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                if (adjustColor)
                {
                    wall3.GetComponent<Renderer>().material = materials.Sample();
                }

                // Create wall 4 (left wall between D-A and H-E)
                var wall4 = m_GameObjectOneWayCache.GetOrInstantiate(prefabs.Sample());
                Vector3 centerDAHE = CalculateOffsetPosition((D + A) / 2, (H + E) / 2);
                wall4.transform.position = centerDAHE + Vector3.right * positionOffset;
                wall4.transform.rotation = Quaternion.Euler(0, 90 - FoV / 2 + angleOffset, 0);
                wall4.transform.localScale = new Vector3(sideWallSize, 100, sideWallWidth);
                Renderer renderer4 = wall4.GetComponent<Renderer>();
                renderer4.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                if (adjustColor)
                {
                    wall4.GetComponent<Renderer>().material = materials.Sample();
                }
            }
        } 
    }

    /// <summary>
    /// Hides all foreground objects after each Scenario Iteration is complete
    /// </summary>
    protected override void OnIterationEnd()
    {
        m_GameObjectOneWayCache.ResetAllObjects();
    }
}
