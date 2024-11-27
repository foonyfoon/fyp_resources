import os
import pickle
import tkinter as tk
from tkinter import ttk
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random


class TreeGUI:
    def __init__(self, root, tree):
        self.root = root
        self.tree = tree
        self.root.title("Tree Interaction GUI")

        # Create a frame for the tree visualization
        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas for the tree visualization
        self.canvas = tk.Canvas(self.frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a scrollbar for the canvas
        self.scrollbar = ttk.Scrollbar(
            self.frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Create a frame inside the canvas
        self.canvas_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.canvas_frame, anchor="nw")

        # Bind the configure event to update the scroll region
        self.canvas_frame.bind("<Configure>", self.on_frame_configure)

        # Create a button to visualize the tree
        self.visualize_button = ttk.Button(
            root, text="Visualize Tree", command=self.visualize_tree
        )
        self.visualize_button.pack(pady=10)

        # Create a text box for displaying node attributes
        self.text_box = tk.Text(root, height=10, width=50)
        self.text_box.pack(pady=10)

        # Create a frame for the attribute buttons
        self.button_frame = ttk.Frame(root)
        self.button_frame.pack(pady=10)

        # Create buttons for displaying individual attributes
        self.attribute_buttons = []

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def hierarchy_pos(self, G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        '''
        From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
        Licensed under Creative Commons Attribution-Share Alike
        '''
        if not nx.is_tree(G):
            raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
            else:
                root = random.choice(list(G.nodes))

        def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):

            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if len(children) != 0:
                dx = width / len(children)
                nextx = xcenter - width / 2 - dx / 2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                         vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                         pos=pos, parent=root)
            return pos

        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    def visualize_tree(self):
        edges, node_types = self.tree.to_edges()
        G = nx.DiGraph(edges)

        color_map = []
        for node in G:
            if node_types[node] == "semantic":
                color_map.append("blue")
            elif node_types[node] == "syntactic":
                color_map.append("green")
            else:
                color_map.append("red")

        # Use the 'spring_layout' for an approximate hierarchical layout
        pos = self.hierarchy_pos(G, self.tree.root.id)
        fig = Figure(figsize=(15, 15))
        ax = fig.add_subplot(111)

        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax)

        # Draw nodes using scatter plot to make them pickable
        for node, color in zip(G.nodes, color_map):
            ax.scatter(*pos[node], c=color, s=100, picker=True, label=node)

        # # Draw labels
        # for node, (x, y) in pos.items():
        #     ax.text(x, y, node, fontsize=12)

        # Display the plot in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Add a click event to display node attributes
        canvas.mpl_connect("pick_event", self.on_click)

    def on_click(self, event):
        # Get the node that was clicked
        clicked_node = event.artist.get_label()

        # Display the attributes of the clicked node in the text box
        if clicked_node is not None:
            node = self.tree.get_node_by_id(clicked_node)
            attributes = vars(node)

            # Clear the text box
            self.text_box.delete("1.0", tk.END)

            # Remove old attribute buttons
            for button in self.attribute_buttons:
                button.destroy()
            self.attribute_buttons.clear()

            # Create new attribute buttons
            for attribute, value in attributes.items():
                button = ttk.Button(
                    self.button_frame,
                    text=f"Show {attribute}",
                    command=lambda value=value: self.show_attribute(value),
                )
                button.pack(side=tk.LEFT, padx=2)
                self.attribute_buttons.append(button)

    def show_attribute(self, value):
        # Clear the text box and insert the attribute value
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, str(value))
