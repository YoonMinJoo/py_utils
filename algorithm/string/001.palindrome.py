# https://leetcode.com/problems/valid-palindrome/

# palindrome인지 반환한다.
# palindrome 확인 방법
# 1. 소문자로 변환
# 2. 알파벳, 숫자만 남김
# 3. 뒤집어서 같은지 확인

class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # 1. 소문자로 변환
        s = s.lower()

        # 2. 알파벳, 숫자만 남김
        import re
        s = re.sub(r'\W', '', s)

        # r을 붙인 이유?
        # \가 이스케이프 처리되지 않도록 원시문자열로 입력
        # 대부분 정규식에서는 붙이는게 좋음 

        # 3. 뒤집어서 비교
        return s == s[::-1]