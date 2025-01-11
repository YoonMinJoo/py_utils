var isPalindrome = function (s) {
  // 1. 소문자 변환
  s = s.toLowerCase();

  // 2. 알파벳, 숫자만 남김
  s = s.replace(/[^a-z0-9]/g, "");

  // 3. 회문 검사
  return s === s.split("").reverse().join("");
};
