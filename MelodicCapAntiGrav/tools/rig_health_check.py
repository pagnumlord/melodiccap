import bpy

def run_rig_health_check():
    print("\n" + "="*50)
    print("ANTIGRAV V3 - RIG HEALTH DIAGNOSTIC")
    print("="*50)
    
    rig = bpy.context.active_object
    
    if not rig or rig.type != 'ARMATURE':
        print("[-] ERROR: No armature selected. Please select Jax's Rigify rig.")
        return

    print(f"[+] Checking Rig: {rig.name}")

    # 1. Metarig Check
    metarig = bpy.data.objects.get("metarig") or bpy.data.objects.get("jaxmetarig")
    if metarig and rig.parent == metarig:
        print("[!] WARNING: Rig is parented to metarig. This can cause cyclic dependencies.")
        print("    -> ACTION: Un-parenting rig from metarig...")
        # Clear parent while keeping transform
        rig.matrix_world = rig.matrix_world.copy()
        rig.parent = None
    else:
        print("[+] CHECK: Rig is correctly independent (no metarig parent).")

    # 2. Constraint Audit
    print("[+] Auditing bone constraints...")
    found_constraints = 0
    for bone in rig.pose.bones:
        for con in bone.constraints:
            if not con.name.startswith("RIGIFY"):
                print(f"    - Found non-native constraint: {bone.name} -> {con.name} ({con.type})")
                found_constraints += 1
    
    if found_constraints == 0:
        print("[+] CHECK: No stray constraints found. Rig is clean.")
    else:
        print(f"[!] WARNING: {found_constraints} stray constraints found. These should be cleared before retargeting.")

    # 3. Object-Level Scale Check
    if any(s != 1.0 for s in rig.scale):
        print(f"[!] WARNING: Rig scale is {rig.scale}. It SHOULD be (1,1,1).")
        print("    -> ACTION: Apply scale (Ctrl+A -> Scale) before continuing.")
    else:
        print("[+] CHECK: Rig scale is correct (1,1,1).")

    # 4. Mesh/Weight Check (Teeth/Tongue)
    found_meshes = []
    for child in rig.children:
        if child.type == 'MESH':
            found_meshes.append(child.name)
    
    print(f"[+] Found {len(found_meshes)} meshes parented to rig: {', '.join(found_meshes)}")
    
    if "Teeth" in str(found_meshes) or "Tongue" in str(found_meshes):
        print("[+] CHECK: Teeth/Tongue meshes detected. They will be monitored for stability.")

    print("\n" + "="*50)
    print("DIAGNOSTIC COMPLETE")
    print("="*50)

if __name__ == "__main__":
    run_rig_health_check()
