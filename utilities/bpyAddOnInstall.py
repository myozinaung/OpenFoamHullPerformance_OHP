import bpy

addon_zip_path = "print3d_toolbox.zip"
bpy.ops.preferences.addon_install(filepath=addon_zip_path)
bpy.ops.preferences.addon_enable(module='print3d_toolbox')

bpy.ops.wm.save_userpref()

installed_addons = bpy.context.preferences.addons
print("Installed Add-ons:")
for addon in installed_addons:
    print(f"- {addon.module}")