# https://leetcode.com/problems/reorder-data-in-log-files/
# 로그 재정렬
# 문자 로그는 숫자 로그 전에 온다
# 문자 로그는 내용으로 정렬하되, 내용이 같으면 식별자로 정렬
# 숫자 로그는 순서를 유지한다.


class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        letters = []
        digits = []

        for log in logs:
            if log.split()[1].isdigit():
                digits.append(log)
            else:
                letters.append(log)

        letters.sort(key=lambda x:(x.split()[1:], x.split()[0]))
        return letters + digits
