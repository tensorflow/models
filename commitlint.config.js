module.exports = {
  extends: ["@commitlint/config-conventional"],
  rules: {
    // Type must be one of these
    "type-enum": [
      2,
      "always",
      [
        "feat", // New feature
        "fix", // Bug fix
        "docs", // Documentation only changes
        "style", // Changes that don't affect code meaning (formatting, etc)
        "refactor", // Code change that neither fixes a bug nor adds a feature
        "perf", // Performance improvement
        "test", // Adding or updating tests
        "chore", // Changes to build process or auxiliary tools
        "revert", // Revert a previous commit
        "build", // Changes to build system or dependencies
        "ci", // Changes to CI configuration
      ],
    ],

    // Scope is REQUIRED and should be the JIRA ticket (e.g., DEVPRD-3177)
    "scope-empty": [2, "never"],
    "scope-case": [2, "always", "upper-case"],

    // Subject must not be sentence-case, start-case, pascal-case, upper-case
    "subject-case": [
      2,
      "never",
      ["sentence-case", "start-case", "pascal-case", "upper-case"],
    ],

    // Subject must not end with a period
    "subject-full-stop": [2, "never", "."],

    // Subject must not be empty
    "subject-empty": [2, "never"],

    // Type must not be empty
    "type-empty": [2, "never"],

    // Header max length (100 characters)
    "header-max-length": [2, "always", 100],

    // Body max line length (100 characters)
    "body-max-line-length": [2, "always", 100],

    // Footer max line length (100 characters)
    "footer-max-line-length": [2, "always", 100],
  },
};
