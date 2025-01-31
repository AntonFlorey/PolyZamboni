length_units_to_str = {
    "MICROMETERS" : "um",
    "MILLIMETERS" : "mm",
    "CENTIMETERS" : "cm",
    "METERS" : "m",
    "KILOMETERS" : "km",
    "THOU" : "thou",
    "INCHES" : "in",
    "FEET" : "ft",
    "MILES" : "mi"
}

unit_to_cm_conversion_table = { # bpy.utils.units is rubbish 
    "MICROMETERS" : 0.0001,
    "MILLIMETERS" : 0.1,
    "CENTIMETERS" : 1,
    "METERS" : 100,
    "KILOMETERS" : 100000,
    "THOU" : 0.00254,
    "INCHES" : 2.54,
    "FEET" : 30.48,
    "MILES" : 160934.4,
}