#　じゃんけんゲームを参考に、簡単なゲームかプログラムを書いてみてください。
import random

PlayerSuccess = 0
AISuccess = 0
HandSelect = ["グー","チョキ","パー"]
Direction = ["→","←","↑","↓"]
Win = 0

def direction_game():
    print("あっちむいてほい(0:→, 1：←, 2:↑, 3:↓)")
    PlayerDirection = int(input())
    AIDirection= random.randint(0,3)
    
    if PlayerDirection == AIDirection:
        print("当たり")
        Success = 1
    else:
        print("はずれ")
        Success = 0
    
    return Success

print("あっちむいてほいゲーム")
print("ルール：先に３回成功させたら勝ち")

while True:
    print("今の成功回数: あなた：{0} 回| AI: {1} 回".format(PlayerSuccess,AISuccess))
    print("選択してください。(0:グー, 1：チョキ, 2:パー)")
    PlayerHand = int(input())
    AIHand= random.randint(0,2)
    print("あなたは{0},AIは{1}を選択しました。".format(HandSelect[PlayerHand],HandSelect[AIHand]))

    ##　勝敗判定
    if PlayerHand == AIHand:
        Win = 0
    elif PlayerHand == 0:
        if AIHand == 1:
            Win = 1
        else:
            Win = -1
    elif PlayerHand == 1:
        if AIHand == 0:
            Win = -1
        else:
            Win = 1
    elif PlayerHand == 2:
        if AIHand == 0:
            Win = 1
        else:
            Win = -1
    ## 得点
    if Win == 0:
        print("引き分けです。")
    elif Win == 1:
        print("あなたの勝ちです。")
        PlayerSuccess += direction_game()
    else:
        print("AIの勝ちです。")
        AISuccess += direction_game()

    ##　終了判定
    if PlayerSuccess >= 2:
        print("あなたが先に２回成功しました。あなたの勝利です。")
        break
    elif AISuccess >= 2:
        print("AIが先に２回成功しました。AIの勝利です。")
        break
