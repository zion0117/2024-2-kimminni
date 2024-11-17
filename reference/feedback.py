import os
from langchain_community.chat_models import ChatOpenAI  # Deprecation 경고 해결
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# ChatGPT API를 이용한 피드백 생성 클래스
class SeniorFriendlyFitnessBot:
    def __init__(self):
        self.prompt = self.create_prompt()
        self.model = self.setup_model()

    def setup_model(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("환경변수에 'OPENAI_API_KEY'가 설정되지 않았습니다.")

        # ChatOpenAI 모델 설정
        chat_model = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=openai_api_key,
            temperature=0.5
        )

        # Conversation 설정
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=chat_model,
            memory=memory
        )
        return conversation

    def create_prompt(self):
        # 올바르게 ChatPromptTemplate 설정
        template = """
        너는 시니어 운동자세교정 서비스의 가상캐릭터 "미니"야.
        운동의 전체 자세 점수가 입력되면 너의 기능을 적극적으로 활용해 텍스트 피드백을 제공해주면 돼. 
        너의 목표는 어르신들이 정확하고 안전한 자세로 운동할 수 있도록 피드백을 제공해주는 거야.
        
        예를 들어 이런식으로 피드백해주면 돼:
        1. 90점 이상: "말 잘하고 계세요! 그대로 하시면 돼요!"
        2. 80점 이상 90점 미만: "조금만 더 조심해볼까요? 무릎이나 허리 각도를 살짝 조정해보세요!"
        3. 80점 미만: "지금 자세가 조금 위험할 수 있어요. 천천히 다시 해보면 좋을 것 같아요. 다치지 않게 조심조심!"
        
        너의 피드백 스타일은 다음과 같았으면 좋겠어:
        1. 자세를 정확하게 교정해준다. 위험하지 않게 안내해준다.
        2. 혼자 운동하는 것에 대한 외로움을 느끼지 않도록 손자, 손녀처럼 친절하고 다정하게 말한다.
        3. 어르신들도 이해할 수 있도록 쉽고 간단하게 답한다.
        
        Human: {input}
        미니:"""

        return ChatPromptTemplate.from_template(template)  # from_template 사용

    def get_feedback(self, comparison_value: str):
        return self.model.run(input=comparison_value)

# 실행 코드
if __name__ == "__main__":
    bot = SeniorFriendlyFitnessBot()

    while True:
        similarity_score = input("운동 자세 점수를 입력하세요 (종료하려면 'quit' 입력): ")
        if similarity_score.lower() == "quit":
            break

        try:
            score = int(similarity_score)
            feedback = bot.get_feedback(f"자세 점수: {score}")
            print("미니 >>", feedback)
        except ValueError:
            print("유효한 숫자 점수를 입력해주세요.")
