import yaml
import logging
import pandas as pd
import psycopg2
from psycopg2 import extras

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostgresConnector:
    def __init__(self, db_params: dict):
        if not db_params:
            raise ValueError("데이터베이스 설정 정보(db_params)가 필요합니다.")
        
        self.db_params = db_params
        self.connection = None
        self.cursor = None
        logger.info("커넥터가 설정 정보와 함께 초기화됨.")

    def connect(self):
        if self.connection and not self.connection.closed:
            logger.info("이미 데이터베이스에 연결됨.")
            return

        try:
            logger.info(f"{self.db_params['host']}:{self.db_params['port']} 데이터베이스에 연결을 시도합니다.")
            self.connection = psycopg2.connect(**self.db_params)
            self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            logger.info("데이터베이스 연결 성공.")
        except psycopg2.OperationalError as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            self.connection = None
            self.cursor = None
            raise

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("데이터베이스 연결 종료")
        self.connection = None
        self.cursor = None

    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        if not self.connection or self.connection.closed:
            logger.warning("데이터베이스에 연결되지 않음. 연결 재시도.")
            self.connect()

        try:
            logger.info(f"쿼리 실행: {query}")
            self.cursor.execute(query, params)
            
            if self.cursor.description:
                columns = [desc[0] for desc in self.cursor.description]
                results = self.cursor.fetchall()
                df = pd.DataFrame(results, columns=columns)
                logger.info(f"{len(df)}개의 레코드를 성공적으로 조회했습니다.")
                return df
            else:
                self.connection.commit()
                logger.info(f"쿼리가 성공적으로 실행되었습니다. 영향 받은 행 수: {self.cursor.rowcount}")
                return pd.DataFrame()

        except psycopg2.Error as e:
            logger.error(f"쿼리 실행 중 오류 발생: {e}")
            if self.connection:
                self.connection.rollback()
            return pd.DataFrame()

if __name__ == '__main__':
    try:
        db_connector = PostgresConnector(config_path='config.yaml')
        db_connector.connect()
        concept_name = 'Hypertension'

        sql_query = """
        SELECT COUNT(DISTINCT p.person_id) AS patient_count
        FROM person p
        JOIN condition_occurrence co ON p.person_id = co.person_id
        JOIN concept c ON co.condition_concept_id = c.concept_id
        WHERE c.concept_name = %s;
        """

        query_result_df = db_connector.execute_query(sql_query, (concept_name,))

        if not query_result_df.empty:
            patient_count = query_result_df['patient_count'].iloc[0]
            print(f"'{concept_name}' 진단 환자 수: {patient_count} 명")

        min_age, max_age, gender_concept = 50, 59, 'MALE'
        sql_query_complex = """
        SELECT COUNT(DISTINCT p.person_id) AS patient_count
        FROM person p
        WHERE (EXTRACT(YEAR FROM CURRENT_DATE) - p.year_of_birth) BETWEEN %s AND %s
          AND p.gender_concept_id = (SELECT concept_id FROM concept WHERE concept_name = %s);
        """
        result_df_complex = db_connector.execute_query(sql_query_complex, (min_age, max_age, gender_concept))
        
        if not result_df_complex.empty:
            print(f"{min_age}~{max_age}세 {gender_concept} 환자 수: {result_df_complex['patient_count'].iloc[0]} 명")

    except Exception as e:
        logger.error(f"메인 실행 중 오류 발생: {e}")

    finally:
        if 'db_connector' in locals() and db_connector.connection:
            db_connector.close()
