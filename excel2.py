import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
import getpass
from dotenv import load_dotenv
load_dotenv()
# OpenAI API 키 설정
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Api key for OPENAI")


# Excel 파일 로드 및 결합 함수
def load_and_merge_excel(file_path):
    """
    두 시트를 읽고 결합된 DataFrame을 반환합니다.
    """
    try:
        # 시트 로드
        df_settlement = pd.read_excel(file_path, sheet_name="하부유통망수수료정산")
        df_policy = pd.read_excel(file_path, sheet_name="SKB정책")

        # 공통 열을 기준으로 결합 (left join)
        merged_df = pd.merge(
            df_settlement,
            df_policy,
            on=["번들결합분류", "인터넷분류", "TV분류"],
            how="left"
        )

        return merged_df
    except FileNotFoundError:
        print(f"Error: {file_path} 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


# 정산 질문을 처리하는 함수
def process_settlement_query(df, query):
    """
    Pandas DataFrame Agent를 사용해 정산 관련 질문을 처리합니다.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )
    response = agent.run(query)
    return response


# 메인 실행
if __name__ == "__main__":
    # Excel 파일 경로
    file_path = "red.xlsx"

    # Excel 파일 로드 및 결합
    merged_df = load_and_merge_excel(file_path)
    if merged_df is None:
        exit()

    # 결합된 DataFrame 확인
    print("Merged DataFrame:")
    print(merged_df.head())

    # 정산 관련 질문
    queries = [
        "업체별로 '실적'과 '총합'을 곱해서 수수료를 계산해줘.",
        "업체별 수수료 합계를 구하고, 가장 높은 업체를 알려줘.",
        "'실적'이 5 이상인 업체의 수수료만 계산해줘."
    ]

    # 각 질문에 대해 정산 처리
    for query in queries:
        print(f"\n질문: {query}")
        result = process_settlement_query(merged_df, query)
        print(f"답변: {result}")

    # 결과를 새 Excel 파일로 저장
    merged_df["수수료"] = merged_df["실적"] * merged_df["총합"]  # 수수료 열 추가
    output_file = "settlement_result.xlsx"
    merged_df.to_excel(output_file, index=False)
    print(f"\n정산 결과가 {output_file}에 저장되었습니다.")
