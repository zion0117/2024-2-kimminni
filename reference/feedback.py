import os
from langchain_openai import ChatOpenAI  # 최신 LangChain OpenAI 패키지 사용
from langchain.prompts import ChatPromptTemplate  # 올바른 경로
from langchain.memory import ConversationBufferMemory  # 올바른 메모리 클래스 경로
from langchain.chains import ConversationChain  # 대화 체인을 사용

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
            model="gpt-3.5-turbo",  # 모델 이름을 안전하게 설정
            openai_api_key=openai_api_key,
            temperature=0.5
        )

        # ConversationBufferMemory로 메모리 설정
        memory = ConversationBufferMemory()  # 간단하고 안정적인 메모리 클래스 사용
        conversation = ConversationChain(  # 대화 체인으로 설정
            llm=chat_model,
            memory=memory
        )
        return conversation

    def create_prompt(self):
        template = """
        너는 시니어 운동자세교정 서비스의 가상캐릭터 "미니"야.
        운동의 전체 자세 점수가 입력되면 너의 기능을 적극적으로 활용해 텍스트 피드백을 제공해줘. 
        어르신들이 정확하고 안전한 자세로 운동할 수 있도록 피드백을 제공하는 게 너의 목표야.
        
        피드백 예시:
        1. 90점 이상: "정말 잘하고 계세요! 그대로 하시면 됩니다! 😊"
        2. 80점 이상 90점 미만: "좋습니다! 하지만 무릎이나 허리 각도를 살짝 조정해보시면 더 좋아질 거예요. 😊"
        3. 80점 미만: "조금 위험할 수 있어요. 천천히 다시 해보시고 다치지 않게 조심하세요. 😥"

        Human: {input}
        미니:"""

        return ChatPromptTemplate.from_template(template)

    def get_feedback(self, comparison_value: str):
        # .predict()를 사용하여 대화 체인을 호출
        return self.model.predict(input=comparison_value)

# 실행 코드
if __name__ == "__main__":
    bot = SeniorFriendlyFitnessBot()

    while True:
        # 사용자로부터 자세 점수를 입력받음
        similarity_score = input("운동 자세 점수를 입력하세요 (종료하려면 'quit' 입력): ")
        if similarity_score.lower() == "quit":
            break

        try:
            # 점수를 정수로 변환
            score = int(similarity_score)
            feedback = bot.get_feedback(f"자세 점수: {score}")
            print("미니 >>", feedback)
        except ValueError:
            print("유효한 숫자 점수를 입력해주세요.")
