"""Blender script: display a vertex-colored PLY with its imported color layer."""

import bpy


objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
if not objects and bpy.context.active_object and bpy.context.active_object.type == "MESH":
    objects = [bpy.context.active_object]
if not objects:
    raise RuntimeError("Select one or more imported PLY meshes before running this script")

for obj in objects:
    color_attributes = obj.data.color_attributes
    if not color_attributes:
        print(f"Skipping {obj.name}: no vertex-color attribute")
        continue
    attribute = color_attributes.active_color or color_attributes.active or color_attributes[0]
    material = bpy.data.materials.new(f"{obj.name} vertex colors")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    try:
        color = nodes.new("ShaderNodeVertexColor")
        color.layer_name = attribute.name
    except RuntimeError:
        color = nodes.new("ShaderNodeAttribute")
        color.attribute_name = attribute.name
    shader.inputs["Roughness"].default_value = 0.72
    links.new(color.outputs["Color"], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(material)
    print(f"{obj.name}: displaying {attribute.name}")
for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        area.spaces.active.shading.type = "MATERIAL"
        break
