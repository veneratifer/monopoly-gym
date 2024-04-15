import numpy as np
import tkinter as tk
from .fields import Purchasable, Property


class GameBoard(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('Monopoly Simulator')
        self.geometry(f"1536x864")
        self.resizable(False, False)
        self.bg_color = '#1e1e1e'
        self.gray_color = '#646464'
        self.configure(bg=self.bg_color)
        self.FIELD_WIDTH = 100
        self.FIELD_HEIGTH = 60
        # variable for game flow control
        self.stop_game = tk.BooleanVar(self, value=True)
        board_frame = tk.Frame(self, bg=self.bg_color)
        board_frame.grid_columnconfigure(tuple(range(11)), weight=1)
        board_frame.grid_rowconfigure(tuple(range(11)), weight=1)
        board_frame.place(x=200, y=40, width=1150, height=710, anchor='nw', in_=self)
        self.board_frame = board_frame
        left_sidebar = tk.Frame(self, bg=self.bg_color, width=200, height=710)
        left_sidebar.place(x=0, y=40)
        self.left_sidebar = left_sidebar
        right_sidebar = tk.Frame(self, bg=self.bg_color, width=150, height=710)
        right_sidebar.place(x=1350, y=40)
        self.right_sidebar = right_sidebar
        self.colors_map = {
            'brown': '#543d0d',
            'lightblue': '#6da1c9',
            'purple': '#7b4596',
            'orange': '#9c8328',
            'red': '#a63030',
            'yellow': '#b3b524',
            'green': '#00570e',
            'blue': '#1c188c'
        }
        # pink, cyan, green, blue
        self.players_colors = ['#c76dbf', '#008080', '#80bd79', '#5959a6']
        get_vars = lambda: [tk.StringVar(self, value='') for _ in range(4)]
        self.players_details = {
            'cash': get_vars(),
            'num_houses': get_vars(),
            'num_properties': get_vars(),
            'num_hotels': get_vars()
        }
        col_vec = np.zeros((40,), dtype=int)
        col_vec[:11] = -1
        col_vec[21:31] = 1
        col_vec[0] = 10
        row_vec = np.zeros((40,), dtype=int)
        row_vec[11:21] = -1
        row_vec[31:] = 1
        row_vec[0] = 10
        # (col, row) coordinates for each of 40 fields
        self.grid_pos_array = zip(row_vec.cumsum(), col_vec.cumsum())
        self.canvases = []
        self.inited = False

    def set_gamemaster(self, gamemaster):
        self.gamemaster = gamemaster
        self.fields = gamemaster.board.fields

    def init_elems(self):
        """
        Initialize all GUI elements
        """
        is_hybrid = [player.hybrid for player in self.gamemaster.players]
        self.players_details['is_hybrid'] = is_hybrid
        self.init_board()
        self.init_left_sidebar()
        self.init_right_sidebar()

    def play_stop(self):
        if self.stop_game.get() == False:
            self.stop_game.set(True)
            self.state_info.configure(text='PAUSED')
        else:
            self.stop_game.set(False)
            self.state_info.configure(text='RUN')

    def update_dices(self, roll_result):
        self.dices[0].configure(text=roll_result[0])
        self.dices[1].configure(text=roll_result[1])

    def init_right_sidebar(self):
        self.right_sidebar.grid_columnconfigure(0, weight=1, minsize=150)
        self.right_sidebar.grid_rowconfigure(list(range(11)), weight=1, minsize=64)

        roll = tk.Label(
            self.right_sidebar,
            text='Last roll results',
            width=15,
            height=3,
            foreground='white',
            bg=self.gray_color,
            border=0
        )
        roll.grid(row=0, column=0, sticky='ne', pady=3)

        dice_1 = tk.Label(
            self.right_sidebar, 
            text='', 
            bg=self.gray_color, 
            height=2,
            width=5,
            foreground='white'
        )
        dice_1.grid(row=1, column=0)
        dice_2 = tk.Label(
            self.right_sidebar, 
            text='', 
            width=5,
            height=2,
            bg=self.gray_color, 
            foreground='white'
        )
        dice_2.grid(row=1, column=0, sticky='e')
        self.dices = (dice_1, dice_2)

    def init_left_sidebar(self):
        label = tk.Label(
            self.left_sidebar, 
            text='Player details',
            foreground='white',
            bg=self.gray_color,
            border=10
        )
        label.place(x=0, y=3, width=170)

        for i in range(4):
            player_tab = tk.Frame(
                self.left_sidebar,
                bg=self.players_colors[i]
            )
            players_details = self.players_details
            player_tab.place(x=0, y=165 * i + 60, width=170, height=145)
            name = f"Player no. {i + 1} {'(hybrid)' if players_details['is_hybrid'][i] else ''}"
            head_label = tk.Label(
                player_tab,
                text=name, 
                bg=self.players_colors[i],
                foreground='white',
                font=('', '11', 'bold')
            )
            head_label.pack(anchor='w', padx=5, pady=5)
            cash_label = tk.Label(
                player_tab, 
                textvariable=players_details['cash'][i], 
                foreground='white', 
                bg=self.players_colors[i],
            )
            cash_label.pack(anchor='w', padx=5, pady=3)
            num_prop_label = tk.Label(
                player_tab,
                textvariable=players_details['num_properties'][i],
                foreground='white', 
                bg=self.players_colors[i]
            )
            num_prop_label.pack(anchor='w', padx=5, pady=3)
            num_houses_label = tk.Label(
                player_tab,
                textvariable=players_details['num_houses'][i],
                foreground='white', 
                bg=self.players_colors[i]
            )
            num_houses_label.pack(anchor='w', padx=5, pady=3)
            num_hotels_label = tk.Label(
                player_tab,
                textvariable=players_details['num_hotels'][i],
                foreground='white', 
                bg=self.players_colors[i]
            )
            num_hotels_label.pack(anchor='w', padx=5, pady=3)

    def init_board(self):
        for i, (row, col) in enumerate(self.grid_pos_array):
            f = self.fields[i]
            color = self.colors_map[f.color] if hasattr(f, 'color') else self.gray_color
            canv = tk.Canvas(
                self.board_frame,
                bg=color,
                width=self.FIELD_WIDTH,
                height=self.FIELD_HEIGTH,
                highlightthickness=0
            )
            canv.create_text(
                self.FIELD_WIDTH / 2,
                self.FIELD_HEIGTH / 2,
                width=70,
                text=f.name,
                font=('Prestige Elite Std','10','normal'),
                fill='white',
                justify='center'
            )
            canv.grid(row=row, column=col, padx=3, pady=3)
            self.canvases.append((i, canv))

        play_stop_btn = tk.Button(
            self.board_frame, 
            text='run-stop', 
            command=self.play_stop, 
            bg=self.gray_color,
            foreground='white',
            border=0
        )
        play_stop_btn.grid(row=5, column=5, columnspan=1, rowspan=1, sticky='nwse')
        state_info = tk.Label(self.board_frame, text='PAUSED', foreground='white', bg=self.bg_color)
        state_info.grid(row=4, column=5, columnspan=1, rowspan=1, sticky='nwse')
        self.state_info = state_info

    def update_board(self):
        players_pos = list()
        players_details = self.players_details
        # details display in left sidebar
        for i, player in enumerate(self.gamemaster.players):
            players_pos.append(player.pos)
            players_details['cash'][i].set('current cash: ' + str(player.money))
            players_details['num_properties'][i].set('num properties: ' + str(len(player.properties)))
            num_houses, num_hotels = 0, 0
            for f_id in player.properties:
                field = self.gamemaster.get_field(f_id)
                if isinstance(field, Property):
                    num_houses += field.build_level
                    if field.build_level == 5:
                        num_houses -= 1
                        num_hotels += 1
            players_details['num_houses'][i].set('num houses: ' + str(num_houses))
            players_details['num_hotels'][i].set('num hotels: ' + str(num_hotels))
        players_pos = np.array(players_pos, dtype=int)

        # game board update
        for loc_id, canv in self.canvases:
            f = self.fields[loc_id]
            # clear canvas except of the text item
            for item_id in canv.find_all():
                if item_id == 1:
                    continue
                canv.delete(item_id)
            for j, player_id in enumerate(np.argwhere(players_pos == loc_id)):
                    player_color = self.players_colors[player_id.item()]
                    x_offset = 10 + j * 15
                    # draw player position indicator
                    canv.create_oval(x_offset, 10, x_offset + 10, 20, fill=player_color, outline="white")
            if isinstance(f, Purchasable):
                if f.owner != None:
                    # draw square associated with ownership
                    player_color = self.players_colors[f.owner]
                    canv.create_rectangle(85, 5, 95, 15, fill=player_color, outline="white")
            if isinstance(f, Property):
                # draw houses
                if f.build_level < 5:
                    for x in range(f.build_level):
                        x_offset = 10 + x * 15
                        canv.create_rectangle(x_offset, 45, x_offset + 10, 55, fill='white', outline="")
                else:
                    canv.create_rectangle(10, 45, 55, 55, fill='white', outline="")