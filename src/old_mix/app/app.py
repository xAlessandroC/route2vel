import sys
import os
from pathlib import Path
print(os.path.join(Path(os.path.abspath(__file__)).parent.parent, "route2vel"))
sys.path.append(os.path.join(Path(os.path.abspath(__file__)).parent.parent))

import tkinter
import tkintermapview
import re
import route2vel
from enum import Enum
from route2vel.postprocess import calc_curvature, interp_gdf_to_csv

from tkinter import Entry, Label, LabelFrame, Frame, Button, Radiobutton, StringVar

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600

START_LATITUDE = 44.494641 
START_LONGITUDE = 11.342279

# App state
class State(Enum):
    START = 1
    STOP = 2
    INTERMEDIATE = 3
    PATH = 4

app_state = State.START

# Map widgets
map_widget = None

# Path selection
starting_point = None
ending_point = None
intermediate_points = []
active_path = None
sampling_text = None
sampling_distance_meters = 5

def submit_task(starting_point, ending_point):
    starting_position = "{}, {}".format(starting_point.position[0], starting_point.position[1])
    ending_position = "{}, {}".format(ending_point.position[0], ending_point.position[1])
    print(starting_position)
    print(ending_position)
    graph_name = re.sub(r'[^\w_. -]', '_', f"{str(starting_position).lower().strip()}-{str(ending_position).lower().strip()}")
    route_dir = route2vel.find_route_osrm([starting_position, ending_position], load_graph=True, load_graph_name=graph_name)
    inverted_geo = [(x[1], x[0]) for x in route_dir.geometry]
    print("-------------------")
    interp_dir = route2vel.interp_from_route(route_dir)
    print(interp_dir)
    print("-------------------")
    route2vel.utils.debug = True
    sampled_gdf = interp_dir.get_points_with_density(
        int(sampling_text.get()), 
        return_gdf=True,
        in_meters=True, 
        gdf_columns=['base_idx', 'junction', 'speed_kph'] if 'junction' in interp_dir.split_gdf.columns else ['base_idx','speed_kph'] #XXX: fix juncture missing,
    )
    print(sampled_gdf)
    print("-------------------")

    calc_curvature(sampled_gdf)

    csv_path = "route_output_full.csv"
    csv_path = os.path.abspath(csv_path)

    interp_gdf_to_csv(
        sampled_gdf, csv_path, 
        separate_roundabout=True if 'junction' in interp_dir.split_gdf.columns else False,  #XXX: fix juncture missing, 
        add_tract_start_col=True, 
        extra_cols=['speed_kph', 'curvature'],
    )

    return inverted_geo


def clearMarkers():
    global app_state, starting_point, ending_point, active_path, intermediate_points
    print("Current state:", app_state)

    map_widget.delete(starting_point)
    map_widget.delete(ending_point)
    map_widget.delete(active_path)

    for point in intermediate_points:
        map_widget.delete(point)

    starting_point = None
    ending_point = None
    active_path = None
    intermediate_points = []
    sampling_text.set("{}".format(sampling_distance_meters))

    app_state = State.START
    selected.set('start')


def checkButton():
    global app_state
    app_state = State[selected.get().upper()] if app_state != State.PATH else app_state
    print("Current state:", app_state)
   

def left_click_event_on_map(coordinates_tuple):
    global app_state, starting_point, ending_point, intermediate_points
    print("Left click event with coordinates:", coordinates_tuple)
    print("Current state:", app_state)

    if app_state == State.START:
        # Delete previous starting marker
        map_widget.delete(starting_point)

        # Set new starting marker
        starting_point = map_widget.set_marker(*coordinates_tuple, text="Start", marker_color_outside="green", text_color="black", marker_color_circle="green")

    elif app_state == State.STOP:
        # Delete previous ending marker
        map_widget.delete(ending_point)

        # Set new ending marker
        ending_point = map_widget.set_marker(*coordinates_tuple, text="Stop", marker_color_outside="red", text_color="black", marker_color_circle="red")

    elif app_state == State.INTERMEDIATE:
        # Set new intermediate marker
        intermediate_point = map_widget.set_marker(*coordinates_tuple, text="{}".format(len(intermediate_points) + 1), marker_color_outside="orange", text_color="black", marker_color_circle="orange")
        intermediate_points.append(intermediate_point)


def submit():
    global app_state, active_path
    print("Submit clicked")

    if starting_point is not None and ending_point is not None:
        print("Markers selected")
        print("Sampling distance:", sampling_text.get())
        app_state = State.PATH
        total_points = [starting_point] + intermediate_points + [ending_point]
        total_inverted_geo = []
        for i in range(len(total_points) - 1):
            total_inverted_geo += submit_task(total_points[i], total_points[i+1])

        active_path = map_widget.set_path(total_inverted_geo)
        
    else:
        print("Markers not selected")


def validate_sampling_distance(value):
    
    if value.isnumeric() and int(value) > 0:
        return True
    else:
        return False


if __name__ == "__main__":
    # create tkinter window
    root_tk = tkinter.Tk()
    root_tk.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    root_tk.title("map_view_example.py")

    # create map widget
    map_widget = tkintermapview.TkinterMapView(root_tk, width=SCREEN_WIDTH, height=SCREEN_HEIGHT*0.6, corner_radius=0)
    map_widget.set_position(START_LATITUDE, START_LONGITUDE)
    map_widget.set_zoom(14)
    map_widget.add_left_click_map_command(left_click_event_on_map)
    map_widget.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

    map_widget.pack()

    interaction_frame = LabelFrame(root_tk, text = "Point Selection", width=SCREEN_WIDTH, height=SCREEN_HEIGHT*0.4, bg="white", borderwidth = 0, highlightthickness = 0)
    interaction_frame.pack(pady=20, anchor="w")

    sampling_text = StringVar()
    sampling_text.set("{}".format(sampling_distance_meters))
    sampling_label = Label(interaction_frame, text="Sampling Distance (m)")
    sampling_label.grid(row = 1, column = 0)
    sampling_entry = Entry(interaction_frame, bd = 5, textvariable = sampling_text)
    sampling_entry.config(validate="all")
    sampling_entry.grid(row = 1, column = 1)

    selected = StringVar()
    r1 = Radiobutton(interaction_frame, text='Start', value='start', variable=selected, command=checkButton)
    r2 = Radiobutton(interaction_frame, text='Intermediate', value='intermediate', variable=selected, command=checkButton)
    r3 = Radiobutton(interaction_frame, text='Stop', value='stop', variable=selected, command=checkButton)
    r1.grid(row=0, column=0, padx=10, pady=10)
    r2.grid(row=0, column=1, padx=10, pady=10)
    r3.grid(row=0, column=2, padx=10, pady=10)
    selected.set('start')

    button = Button(interaction_frame, text="Clear", command=clearMarkers)
    button.grid(row=2, column=0, padx=10, pady=10)

    button = Button(interaction_frame, text="Submit", command=submit)
    button.grid(row=2, column=1, padx=10, pady=10)

    interaction_frame.columnconfigure(0, weight=1)
    interaction_frame.rowconfigure((0,1), weight=1)

    root_tk.mainloop()