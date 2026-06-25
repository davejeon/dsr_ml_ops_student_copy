import curses
import time
from holy_class import holy_shit

def main(stdscr):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.curs_set(0)

    height, width = stdscr.getmaxyx()
    header = stdscr.subwin(3, width, 0, 0)
    body = stdscr.subwin(height - 3, width, 3, 0)
    text_message = holy_shit().holy()
    messages = []
    for step in range(10):
        header.erase()
        header.bkgd(" ", curses.color_pair(1))
        title = f"Dashboard  |  Freaking out step: {step}"
        header.addstr(1, 2, title, curses.color_pair(1) | curses.A_BOLD)
        header.refresh()

        messages.append(f"[{step:02d}] {text_message}")
        body.erase()
        body.box()
        for i, msg in enumerate(messages[-(height - 5):]):
            body.addstr(i + 1, 2, msg, curses.color_pair(2))
        body.refresh()

        time.sleep(1)

    stdscr.getch()

curses.wrapper(main)