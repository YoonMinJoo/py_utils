var reorderLogFiles = function (logs) {
  const letters = [];
  const digits = [];

  for (let log of logs) {
    let parts = log.split(" ");
    if (/\d/.test(parts[1])) {
      digits.push(log);
    } else {
      letters.push(log);
    }
  }

  letters.sort((a, b) => {
    const aParts = a.split(" ").slice(1).join(" ");
    const bParts = b.split(" ").slice(1).join(" ");

    if (aParts === bParts) {
      return a.split(" ")[0].localeCompare(b.split(" ")[0]);
    } else {
      return aParts.localeCompare(bParts);
    }
  });
  return letters.concat(digits);
};
