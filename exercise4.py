import random
import re
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
import plotly.express as px
from colored import Fore, Style

WIN = 1
DRAW = 0
LOSE = -1
HIT = 0
STAY = 1


def print_colored(text, *args):
    text = text.replace("Player", f"{Fore.green}Player{Style.reset}")
    text = text.replace("Dealer", f"{Fore.rgb(255,124,198)}Dealer{Style.reset}")
    # print(text, *args)


class Deck:
    def __init__(self):
        self.cards = [i for i in range(2, 11)] * 4
        self.cards += [10] * 12
        self.cards += [11] * 4
        random.shuffle(self.cards)
        self.index = 0

    def deal(self):
        card = self.cards[self.index]
        self.index += 1
        return card


class Player:
    def __init__(self):
        self.cards = []
        self.dealer_card = None
        self.value = 0

    def load_entry(self, file_name: str):
        df = pd.read_csv(file_name)
        for _, row in df.iterrows():
            self.entry[(row["Ace"], row["Value"], row["Dealer"]), HIT] = row["Hit"], 100
            self.entry[(row["Ace"], row["Value"], row["Dealer"]), STAY] = (
                row["Stay"],
                100,
            )

    def reset(self):
        self.cards = []
        self.dealer_card = None
        self.value = 0

    def add_card(self, card):
        self.cards.append(card)
        self.ace_count = self.cards.count(11)
        self.tmp_value = sum([i for i in self.cards])
        while self.tmp_value > 21 and self.ace_count > 0:
            self.tmp_value -= 10
            self.ace_count -= 1
        self.value = self.tmp_value

    def open_card(self):
        return self.cards[0]

    def see(self, card):
        self.dealer_card = card

    def policy(self):
        raise NotImplementedError

    def get_state(self):
        return (
            self.cards.count(11),
            sum([i for i in self.cards if i != 11]),
            self.dealer_card,
        )

    def receive_result(self, result):
        pass

    def save_entry(self, file_name: str):
        data = []
        state_set = set([i[0] for i in self.entry.keys()])
        for state in sorted(state_set, key=lambda x: (x[0], x[1], x[2])):
            q_hit, _ = self.entry.get((state, HIT), (0, 0))
            q_stay, _ = self.entry.get((state, STAY), (0, 0))
            data.append(
                [state[0], state[1], state[2], round(q_hit, 3), round(q_stay, 3)]
            )
        df = pd.DataFrame(data, columns=["Ace", "Value", "Dealer", "Hit", "Stay"])
        df.to_csv(f"files/4_{file_name}.csv", index=False)

    def __repr__(self):
        return f"{self.cards} ({self.value})"

    def __str__(self):
        return f"{self.__class__.__name__}"


class PlayerBase(Player):
    def policy(self):
        if self.value < 17:
            return HIT
        else:
            return STAY

    def save_entry(self, file_name: str):
        pass


class PlayerMC(Player):
    def __init__(self, file_name: str = None):
        super().__init__()
        self.entry = {}
        if file_name:
            self.load_entry(file_name)
        self.action_history = []

    def policy(self):
        state = self.get_state()
        q_hit, _ = self.entry.get((state, HIT), (0, 0))
        q_stay, _ = self.entry.get((state, STAY), (0, 0))
        if q_hit > q_stay:
            next_action = HIT
        elif q_hit < q_stay:
            next_action = STAY
        else:
            next_action = random.choice([HIT, STAY])

        self.action_history.append((state, next_action))
        return next_action

    def receive_result(self, result):
        for state, action in self.action_history:
            q, episode_count = self.entry.get((state, action), (0, 0))
            episode_count += 1
            self.entry[(state, action)] = (
                (q * (episode_count - 1) + result) / episode_count,
                episode_count,
            )
        self.action_history = []


class PlayerSARSA(Player):
    def __init__(self, file_name: str = None):
        super().__init__()
        self.entry = {}
        if file_name:
            self.load_entry(file_name)
        self.previous_state = None
        self.previous_action = None
        self.episode_count = 0

    def policy(self):
        state = self.get_state()

        q_hit, _ = self.entry.get((state, HIT), (0, 0))
        q_stay, _ = self.entry.get((state, STAY), (0, 0))
        if q_hit > q_stay:
            next_action = HIT
        elif q_hit < q_stay:
            next_action = STAY
        else:
            next_action = random.choice([HIT, STAY])

        if random.random() < 1 / (self.episode_count + 1):
            next_action = random.choice([HIT, STAY])

        if self.previous_action is not None:  # update TD
            q, episode_count = self.entry.get(
                (self.previous_state, self.previous_action), (0, 0)
            )
            episode_count += 1
            self.entry[(self.previous_state, self.previous_action)] = (
                q
                + 1
                / episode_count
                * (0 + self.entry.get((state, next_action), (0, 0))[0] - q),
                episode_count,
            )

        self.previous_state = state
        self.previous_action = next_action
        return next_action

    def receive_result(self, result):
        state = self.get_state()
        q, episode_count = self.entry.get(
            (self.previous_state, self.previous_action), (0, 0)
        )
        episode_count += 1
        self.entry[(self.previous_state, self.previous_action)] = (
            q
            + 1
            / episode_count
            * (result + self.entry.get((state, STAY), (0, 0))[0] - q),
            episode_count,
        )


class PlayerQ(Player):
    def __init__(self, file_name: str = None):
        super().__init__()
        self.entry = {}
        if file_name:
            self.load_entry(file_name)
        self.previous_state = None
        self.previous_action = None
        self.episode_count = 0

    def policy(self):
        state = self.get_state()

        q_hit, _ = self.entry.get((state, HIT), (0, 0))
        q_stay, _ = self.entry.get((state, STAY), (0, 0))
        if q_hit > q_stay:
            next_action = HIT
        elif q_hit < q_stay:
            next_action = STAY
        else:
            next_action = random.choice([HIT, STAY])

        if self.previous_action is not None:  # update Q
            q, episode_count = self.entry.get(
                (self.previous_state, self.previous_action), (0, 0)
            )
            episode_count += 1
            self.entry[(self.previous_state, self.previous_action)] = (
                q + 1 / episode_count * (0 + max(q_hit, q_stay) - q),
                episode_count,
            )

        self.previous_state = state
        self.previous_action = next_action
        return next_action

    def receive_result(self, result):
        state = self.get_state()
        q, episode_count = self.entry.get(
            (self.previous_state, self.previous_action), (0, 0)
        )
        episode_count += 1
        self.entry[(self.previous_state, self.previous_action)] = (
            q + 1 / episode_count * (result - q),
            episode_count,
        )


class PlayerDQ(Player):
    def __init__(self, file_name: str = None):
        super().__init__()
        self.entry = [{}, {}]
        if file_name:
            self.load_entry(file_name)
        self.previous_state = None
        self.previous_action = None
        self.episode_count = 0

    def policy(self):
        flag = random.choice([0, 1])
        entry = self.entry[flag]
        other_entry = self.entry[1 - flag]
        state = self.get_state()

        q_hit, _ = entry.get((state, HIT), (0, 0))
        q_stay, _ = entry.get((state, STAY), (0, 0))
        other_q_hit, _ = other_entry.get((state, HIT), (0, 0))
        other_q_stay, _ = other_entry.get((state, STAY), (0, 0))
        if q_hit + other_q_hit > q_stay + other_q_stay:
            next_action = HIT
        elif q_hit + other_q_hit < q_stay + other_q_stay:
            next_action = STAY
        else:
            next_action = random.choice([HIT, STAY])
        if q_hit > q_stay:
            best_action = HIT
        elif q_hit < q_stay:
            best_action = STAY
        else:
            best_action = random.choice([HIT, STAY])

        if self.previous_action is not None:  # update Q
            q, episode_count = other_entry.get(
                (self.previous_state, self.previous_action), (0, 0)
            )
            episode_count += 1
            entry[(self.previous_state, self.previous_action)] = (
                q
                + 1
                / episode_count
                * (0 + other_entry.get((state, best_action), (0, 0))[0] - q),
                episode_count,
            )

        self.previous_state = state
        self.previous_action = next_action
        return next_action

    def receive_result(self, result):
        flag = random.choice([0, 1])
        entry = self.entry[flag]
        other_entry = self.entry[1 - flag]

        state = self.get_state()
        q, episode_count = other_entry.get(
            (self.previous_state, self.previous_action), (0, 0)
        )
        episode_count += 1
        entry[(self.previous_state, self.previous_action)] = (
            q + 1 / episode_count * (result - q),
            episode_count,
        )

    def save_entry(self, file_name: str):
        for i, entry in enumerate(self.entry):
            data = []
            state_set = set([i[0] for i in entry.keys()])
            for state in sorted(state_set, key=lambda x: (x[0], x[1], x[2])):
                q_hit, _ = entry.get((state, HIT), (0, 0))
                q_stay, _ = entry.get((state, STAY), (0, 0))
                data.append(
                    [state[0], state[1], state[2], round(q_hit, 3), round(q_stay, 3)]
                )
            df = pd.DataFrame(data, columns=["Ace", "Value", "Dealer", "Hit", "Stay"])
            df.to_csv(f"files/4_{file_name}_{i}.csv", index=False)


class PlayerUser(Player):
    def policy(self):
        print_colored("Dealer's card: ", self.dealer_card)
        action = input("Hit or stay? (h/s): ")
        if action == "h":
            return HIT
        else:
            return STAY


class Dealer(Player):
    def add_card(self, card):
        self.cards.append(card)
        self.value += card

    def policy(self):
        if self.value < 17:
            return HIT
        else:
            return STAY


class BlackJack:
    def __init__(self, dealer: Player, player: Player):
        self.deck = Deck()
        self.dealer = dealer
        self.player = player
        self.player.add_card(self.deck.deal())
        self.dealer.add_card(self.deck.deal())
        self.player.add_card(self.deck.deal())
        self.dealer.add_card(self.deck.deal())
        self.player.see(self.dealer.open_card())

    def play(self):
        while True:
            print_colored("Player: ", repr(self.player))
            print_colored("Dealer: ", repr(self.dealer))

            if self.player.value > 21:
                print_colored("Player busts")
                return LOSE
            elif self.dealer.value > 21:
                print_colored("Dealer busts")
                return WIN
            elif self.player.value == 21:
                print_colored("Player wins")
                return WIN
            elif self.dealer.value == 21:
                print_colored("Dealer wins")
                return LOSE
            else:
                print_colored("Player's turn")
                action = self.player.policy()

                if action == STAY:
                    print_colored("Player stands")
                    print_colored("Dealer's turn")
                    while True:
                        dealer_action = self.dealer.policy()
                        if dealer_action == STAY:
                            print_colored("Dealer stands")
                            break
                        new_card = self.deck.deal()
                        print_colored("Dealer gets ", new_card)
                        self.dealer.add_card(new_card)
                        print_colored("Dealer: ", repr(self.dealer))
                    if self.dealer.value > 21:
                        print_colored("Dealer busts")
                        return WIN
                    elif self.dealer.value > self.player.value:
                        print_colored("Dealer wins")
                        return LOSE
                    elif self.dealer.value == self.player.value:
                        print_colored("Draw")
                        return DRAW
                    else:
                        print_colored("Player wins")
                        return WIN
                elif action == HIT:
                    print_colored("Player hits")
                    new_card = self.deck.deal()
                    print_colored("Player gets ", new_card)
                    self.player.add_card(new_card)
                else:
                    raise ValueError("Invalid action")


if __name__ == "__main__":
    num_round = 300
    num_episode_per_round = 1000
    dealer = Dealer()

    # game = BlackJack(dealer, PlayerUser())
    # print(game.play())

    players = [PlayerBase(), PlayerMC(), PlayerSARSA(), PlayerQ(), PlayerDQ()]

    for player in players:
        print(f"===== {player.__class__.__name__} =====")

        win_rate = []
        for round_ in range(num_round):
            win = 0
            for i in range(num_episode_per_round):
                print_colored(f"===== GAME {i+1} =====")
                dealer.reset()
                player.reset()
                game = BlackJack(dealer, player)
                result = game.play()
                player.receive_result(result)
                win += result

            player.save_entry(f"{player.__class__.__name__}_entry")

            win_rate.append(win / num_episode_per_round)
            if (round_ + 1) % 10 == 0:
                print(f"Round: {round_+1} Win rate: {win / num_episode_per_round}")

        fig = px.line(y=win_rate)
        fig.write_image(f"images/4_{player.__class__.__name__}.png")
